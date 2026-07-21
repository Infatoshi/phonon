#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#import <objc/runtime.h>
#import <unistd.h>

static IMP original_new_pipeline;
static IMP original_set_pipeline;
static IMP original_dispatch_threads;
static IMP original_dispatch_groups;
static const void *pipeline_name_key = &pipeline_name_key;
static const void *encoder_name_key = &encoder_name_key;
static const char *profile_flag;
static mach_timebase_info_data_t timebase;

static id new_pipeline(id self, SEL selector, id<MTLFunction> function, NSError **error) {
    id pipeline = ((id (*)(id, SEL, id, NSError **))original_new_pipeline)(
        self, selector, function, error);
    if (pipeline && function.name) {
        objc_setAssociatedObject(
            pipeline, pipeline_name_key, function.name, OBJC_ASSOCIATION_COPY_NONATOMIC);
    }
    return pipeline;
}

static void set_pipeline(id self, SEL selector, id pipeline) {
    ((void (*)(id, SEL, id))original_set_pipeline)(self, selector, pipeline);
    NSString *name = objc_getAssociatedObject(pipeline, pipeline_name_key);
    if (name.length > 0) {
        objc_setAssociatedObject(self, encoder_name_key, name, OBJC_ASSOCIATION_COPY_NONATOMIC);
    }
}

static bool recording(void) {
    return profile_flag && access(profile_flag, F_OK) == 0;
}

static void emit(id encoder, uint64_t elapsed) {
    NSString *name = objc_getAssociatedObject(encoder, encoder_name_key);
    if (name.length == 0) return;
    uint64_t nanoseconds = elapsed * timebase.numer / timebase.denom;
    fprintf(stderr, "PHONON_KERNEL\t%s\t%llu\n", name.UTF8String, nanoseconds);
}

static void dispatch_threads(id self, SEL selector, MTLSize grid, MTLSize group) {
    bool active = recording();
    uint64_t start = active ? mach_continuous_time() : 0;
    ((void (*)(id, SEL, MTLSize, MTLSize))original_dispatch_threads)(
        self, selector, grid, group);
    if (active) emit(self, mach_continuous_time() - start);
}

static void dispatch_groups(id self, SEL selector, MTLSize grid, MTLSize group) {
    bool active = recording();
    uint64_t start = active ? mach_continuous_time() : 0;
    ((void (*)(id, SEL, MTLSize, MTLSize))original_dispatch_groups)(
        self, selector, grid, group);
    if (active) emit(self, mach_continuous_time() - start);
}

__attribute__((constructor)) static void install_hooks(void) {
    @autoreleasepool {
        profile_flag = getenv("PHONON_PROFILE_FLAG");
        mach_timebase_info(&timebase);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        Class device_class = object_getClass(device);
        Method method = class_getInstanceMethod(
            device_class, @selector(newComputePipelineStateWithFunction:error:));
        original_new_pipeline = method_setImplementation(method, (IMP)new_pipeline);

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
        Class encoder_class = object_getClass(encoder);

        method = class_getInstanceMethod(encoder_class, @selector(setComputePipelineState:));
        original_set_pipeline = method_setImplementation(method, (IMP)set_pipeline);
        method = class_getInstanceMethod(
            encoder_class, @selector(dispatchThreads:threadsPerThreadgroup:));
        original_dispatch_threads = method_setImplementation(method, (IMP)dispatch_threads);
        method = class_getInstanceMethod(
            encoder_class, @selector(dispatchThreadgroups:threadsPerThreadgroup:));
        original_dispatch_groups = method_setImplementation(method, (IMP)dispatch_groups);
        [encoder endEncoding];
    }
}
