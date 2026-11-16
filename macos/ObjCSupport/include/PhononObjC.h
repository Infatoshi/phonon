#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Runs `block` and turns an Objective-C exception into an NSError. AVFAudio
/// reports some device-change races (a tap installed in a format the hardware
/// just left) by raising, which Swift cannot catch and which aborts the app.
BOOL PhononCatchException(NS_NOESCAPE void (^block)(void), NSError *_Nullable *_Nullable error);

NS_ASSUME_NONNULL_END
