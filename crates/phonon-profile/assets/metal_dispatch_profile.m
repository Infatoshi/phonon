#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#import <objc/runtime.h>
#import <os/lock.h>
#import <string.h>
#import <unistd.h>

// Concrete Metal classes are private driver classes, so every hook is installed
// on the exact class of a live object rather than on a name we guess. Encoder
// classes are registered lazily: the command buffer is hooked first, and each
// distinct encoder class it hands back is swizzled the first time it is seen.
// That covers MLX, which asks for a concurrent-dispatch encoder, as well as any
// wrapper class (MPS, Metal validation layers) that is not the class of the
// throwaway probe encoder created below.
#define MAX_ENCODER_CLASSES 8

typedef struct {
    Class encoder_class;
    IMP set_pipeline;
    IMP dispatch_threads;
    IMP dispatch_groups;
    IMP dispatch_indirect;
} encoder_hooks;

static encoder_hooks encoders[MAX_ENCODER_CLASSES];
static unsigned encoder_count;
static os_unfair_lock encoder_lock = OS_UNFAIR_LOCK_INIT;

static IMP original_pipeline_function;
static IMP original_pipeline_function_options;
static IMP original_pipeline_descriptor;
static IMP original_compute_encoder;
static IMP original_compute_encoder_dispatch_type;
static IMP original_commit;

static const void *pipeline_name_key = &pipeline_name_key;
static const void *encoder_name_key = &encoder_name_key;
static const char *profile_flag;
static mach_timebase_info_data_t timebase;
static uint64_t commit_serial;

// Replace one method on exactly this class, never on an inherited definition, so
// sibling subclasses keep their own behaviour and the returned original is
// always the implementation this class used to reach.
static IMP swizzle(Class target, SEL selector, IMP replacement) {
    Method method = class_getInstanceMethod(target, selector);
    if (!method) return NULL;
    IMP original = class_replaceMethod(target, selector, replacement,
                                       method_getTypeEncoding(method));
    return original ? original : method_getImplementation(method);
}

static const encoder_hooks *hooks_for(id encoder) {
    Class encoder_class = object_getClass(encoder);
    for (unsigned index = 0; index < encoder_count; index++) {
        if (encoders[index].encoder_class == encoder_class) return &encoders[index];
    }
    return NULL;
}

static bool recording(void) {
    if (!profile_flag || !*profile_flag) return false;
    // The Rust driver passes an absolute path and creates the file only around
    // the measured request. A bare value such as 1 records unconditionally,
    // which is what a manual reproduction command wants.
    if (strchr(profile_flag, '/')) return access(profile_flag, F_OK) == 0;
    return strcmp(profile_flag, "0") != 0;
}

static void remember_name(id pipeline, NSString *name) {
    if (pipeline && name.length > 0) {
        objc_setAssociatedObject(
            pipeline, pipeline_name_key, name, OBJC_ASSOCIATION_COPY_NONATOMIC);
    }
}

static id new_pipeline(id self, SEL selector, id<MTLFunction> function, NSError **error) {
    id pipeline = ((id (*)(id, SEL, id, NSError **))original_pipeline_function)(
        self, selector, function, error);
    remember_name(pipeline, function.name);
    return pipeline;
}

static id new_pipeline_options(id self, SEL selector, id<MTLFunction> function,
                               MTLPipelineOption options, id *reflection, NSError **error) {
    id pipeline = ((id (*)(id, SEL, id, MTLPipelineOption, id *, NSError **))
                       original_pipeline_function_options)(
        self, selector, function, options, reflection, error);
    remember_name(pipeline, function.name);
    return pipeline;
}

static id new_pipeline_descriptor(id self, SEL selector, MTLComputePipelineDescriptor *descriptor,
                                  MTLPipelineOption options, id *reflection, NSError **error) {
    id pipeline = ((id (*)(id, SEL, id, MTLPipelineOption, id *, NSError **))
                       original_pipeline_descriptor)(
        self, selector, descriptor, options, reflection, error);
    NSString *name = descriptor.computeFunction.name;
    remember_name(pipeline, name.length > 0 ? name : descriptor.label);
    return pipeline;
}

static void set_pipeline(id self, SEL selector, id pipeline) {
    const encoder_hooks *hooks = hooks_for(self);
    if (hooks) ((void (*)(id, SEL, id))hooks->set_pipeline)(self, selector, pipeline);
    NSString *name = objc_getAssociatedObject(pipeline, pipeline_name_key);
    if (name.length == 0) name = [(id<MTLComputePipelineState>)pipeline label];
    if (name.length > 0) {
        objc_setAssociatedObject(self, encoder_name_key, name, OBJC_ASSOCIATION_COPY_NONATOMIC);
    }
}

static void emit(id encoder, uint64_t elapsed) {
    NSString *name = objc_getAssociatedObject(encoder, encoder_name_key);
    if (name.length == 0) return;
    uint64_t nanoseconds = elapsed * timebase.numer / timebase.denom;
    fprintf(stderr, "PHONON_KERNEL\t%s\t%llu\n", name.UTF8String, nanoseconds);
}

static void dispatch_threads(id self, SEL selector, MTLSize grid, MTLSize group) {
    const encoder_hooks *hooks = hooks_for(self);
    if (!hooks) return;
    bool active = recording();
    uint64_t start = active ? mach_continuous_time() : 0;
    ((void (*)(id, SEL, MTLSize, MTLSize))hooks->dispatch_threads)(self, selector, grid, group);
    if (active) emit(self, mach_continuous_time() - start);
}

static void dispatch_groups(id self, SEL selector, MTLSize grid, MTLSize group) {
    const encoder_hooks *hooks = hooks_for(self);
    if (!hooks) return;
    bool active = recording();
    uint64_t start = active ? mach_continuous_time() : 0;
    ((void (*)(id, SEL, MTLSize, MTLSize))hooks->dispatch_groups)(self, selector, grid, group);
    if (active) emit(self, mach_continuous_time() - start);
}

static void dispatch_indirect(id self, SEL selector, id<MTLBuffer> buffer, NSUInteger offset,
                              MTLSize group) {
    const encoder_hooks *hooks = hooks_for(self);
    if (!hooks) return;
    bool active = recording();
    uint64_t start = active ? mach_continuous_time() : 0;
    ((void (*)(id, SEL, id, NSUInteger, MTLSize))hooks->dispatch_indirect)(
        self, selector, buffer, offset, group);
    if (active) emit(self, mach_continuous_time() - start);
}

static void register_encoder_class(id encoder) {
    if (!encoder) return;
    Class encoder_class = object_getClass(encoder);
    os_unfair_lock_lock(&encoder_lock);
    bool known = false;
    for (unsigned index = 0; index < encoder_count; index++) {
        if (encoders[index].encoder_class == encoder_class) known = true;
    }
    if (!known && encoder_count < MAX_ENCODER_CLASSES) {
        encoder_hooks hooks = {.encoder_class = encoder_class};
        hooks.set_pipeline = swizzle(
            encoder_class, @selector(setComputePipelineState:), (IMP)set_pipeline);
        hooks.dispatch_threads = swizzle(
            encoder_class, @selector(dispatchThreads:threadsPerThreadgroup:),
            (IMP)dispatch_threads);
        hooks.dispatch_groups = swizzle(
            encoder_class, @selector(dispatchThreadgroups:threadsPerThreadgroup:),
            (IMP)dispatch_groups);
        hooks.dispatch_indirect = swizzle(
            encoder_class,
            @selector(dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:
                                                    threadsPerThreadgroup:),
            (IMP)dispatch_indirect);
        if (hooks.set_pipeline && hooks.dispatch_threads && hooks.dispatch_groups) {
            encoders[encoder_count] = hooks;
            // Publish only after every field is written; readers run lock free.
            __sync_synchronize();
            encoder_count++;
        }
    }
    os_unfair_lock_unlock(&encoder_lock);
}

static id compute_encoder(id self, SEL selector) {
    id encoder = ((id (*)(id, SEL))original_compute_encoder)(self, selector);
    register_encoder_class(encoder);
    return encoder;
}

static id compute_encoder_dispatch_type(id self, SEL selector, MTLDispatchType type) {
    id encoder = ((id (*)(id, SEL, MTLDispatchType))original_compute_encoder_dispatch_type)(
        self, selector, type);
    register_encoder_class(encoder);
    return encoder;
}

// Per dispatch GPU counters need MTLCounterSamplingPointAtDispatchBoundary,
// which Apple silicon does not support, so the per kernel figure above is CPU
// encode time. GPU busy time is still exact per command buffer. PHONON_COMMIT
// is written from the encoding thread, so every PHONON_KERNEL line after the
// previous marker belongs to this buffer, and the asynchronous
// PHONON_GPU_BUFFER line carries the same serial number.
static void commit(id self, SEL selector) {
    if (recording()) {
        uint64_t serial = __sync_add_and_fetch(&commit_serial, 1);
        fprintf(stderr, "PHONON_COMMIT\t%llu\n", serial);
        [(id<MTLCommandBuffer>)self addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            double seconds = buffer.GPUEndTime - buffer.GPUStartTime;
            if (seconds > 0) {
                fprintf(stderr, "PHONON_GPU_BUFFER\t%llu\t%llu\n", serial,
                        (uint64_t)(seconds * 1e9));
            }
        }];
    }
    ((void (*)(id, SEL))original_commit)(self, selector);
}

__attribute__((constructor)) static void install_hooks(void) {
    @autoreleasepool {
        profile_flag = getenv("PHONON_PROFILE_FLAG");
        mach_timebase_info(&timebase);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        Class device_class = object_getClass(device);
        original_pipeline_function = swizzle(
            device_class, @selector(newComputePipelineStateWithFunction:error:),
            (IMP)new_pipeline);
        original_pipeline_function_options = swizzle(
            device_class, @selector(newComputePipelineStateWithFunction:options:reflection:error:),
            (IMP)new_pipeline_options);
        original_pipeline_descriptor = swizzle(
            device_class,
            @selector(newComputePipelineStateWithDescriptor:options:reflection:error:),
            (IMP)new_pipeline_descriptor);

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> buffer = [queue commandBuffer];
        Class buffer_class = object_getClass(buffer);
        original_compute_encoder = swizzle(
            buffer_class, @selector(computeCommandEncoder), (IMP)compute_encoder);
        original_compute_encoder_dispatch_type = swizzle(
            buffer_class, @selector(computeCommandEncoderWithDispatchType:),
            (IMP)compute_encoder_dispatch_type);
        original_commit = swizzle(buffer_class, @selector(commit), (IMP)commit);

        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
        [encoder endEncoding];
    }
}
