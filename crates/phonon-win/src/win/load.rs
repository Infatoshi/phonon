//! How busy the machine's CPUs are, sampled in the background.
//!
//! A dictation that starts while every core is busy sends its correction to the
//! one-thread server (`polish::Corrector::prepare`). The decision is made at
//! key-down, when Phonon itself is idle, from the last half-second of
//! `GetSystemTimes`: reading it there costs nothing and adds no delay.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use windows_sys::Win32::Foundation::FILETIME;
use windows_sys::Win32::System::Threading::GetSystemTimes;

/// Busy fraction of the last interval in thousandths; `u32::MAX` until the first
/// interval has been measured.
static BUSY: AtomicU32 = AtomicU32::new(u32::MAX);

fn ticks(time: &FILETIME) -> u64 {
    (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime)
}

/// (idle, total) in 100 ns units, summed over every CPU. Kernel time includes idle.
fn sample() -> Option<(u64, u64)> {
    let zero = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let (mut idle, mut kernel, mut user) = (zero, zero, zero);
    if unsafe { GetSystemTimes(&mut idle, &mut kernel, &mut user) } == 0 {
        return None;
    }
    Some((ticks(&idle), ticks(&kernel) + ticks(&user)))
}

/// The busy fraction between two samples.
pub fn busy_between(before: (u64, u64), after: (u64, u64)) -> Option<f64> {
    let idle = after.0.checked_sub(before.0)?;
    let total = after.1.checked_sub(before.1)?;
    if total == 0 {
        return None;
    }
    Some((1.0 - idle as f64 / total as f64).clamp(0.0, 1.0))
}

/// Start the sampler thread. Call once.
pub fn start() {
    std::thread::spawn(|| {
        let Some(mut last) = sample() else { return };
        loop {
            std::thread::sleep(Duration::from_millis(500));
            let Some(now) = sample() else { continue };
            if let Some(busy) = busy_between(last, now) {
                BUSY.store((busy * 1000.0) as u32, Ordering::Relaxed);
            }
            last = now;
        }
    });
}

/// The busy fraction of the last half second, once one has been measured.
pub fn busy() -> Option<f64> {
    match BUSY.load(Ordering::Relaxed) {
        u32::MAX => None,
        permille => Some(f64::from(permille) / 1000.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn busy_is_the_share_of_time_not_idle() {
        assert_eq!(busy_between((0, 0), (250, 1000)), Some(0.75));
        assert_eq!(busy_between((100, 100), (100, 1100)), Some(1.0));
        // no time passed, or a counter went backwards: unknown
        assert_eq!(busy_between((0, 500), (0, 500)), None);
        assert_eq!(busy_between((10, 500), (5, 900)), None);
    }

    #[test]
    fn the_sampler_reports_a_fraction() {
        start();
        std::thread::sleep(Duration::from_millis(1200));
        let busy = busy().expect("measured after two intervals");
        assert!((0.0..=1.0).contains(&busy));
    }
}
