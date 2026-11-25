#[cfg(not(target_family = "wasm"))]
fn main() {
    use std::panic::{catch_unwind, resume_unwind};

    if let Some(mode) = std::env::var_os("GPUI_NATIVE_PANIC_TEST") {
        gpui::set_native_panic_observer(|context, payload| {
            let value = payload.downcast_ref::<u32>().expect("original payload");
            eprintln!("observed {context} {value}");
            panic!("observer failed");
        });
        gpui::set_native_panic_observer(|_, _| panic!("replaced original observer"));
        match mode.to_str().unwrap() {
            "baseline" => std::process::abort(),
            "recovered" => {
                let result = gpui::with_native_callback("recovered", || {
                    assert!(catch_unwind(|| panic!("recovered inner panic")).is_err());
                    17
                });
                assert_eq!(result, 17);
            }
            "escape" => gpui::with_native_callback("outer", || {
                gpui::with_native_callback("inner", || {
                    let _ = catch_unwind(|| panic!("old panic"));
                    resume_unwind(Box::new(42_u32));
                });
            }),
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            "foreground" | "work" | "timer" => {
                let platform = gpui_platform::current_platform(true);
                let dispatcher = platform.background_executor().dispatcher().clone();
                std::thread::spawn(|| {
                    std::thread::sleep(std::time::Duration::from_secs(20));
                    std::process::exit(97);
                });
                platform.run(Box::new(move || {
                    let (runnable, task) = async_task::Builder::new()
                        .metadata(gpui::RunnableMeta::new_with_callers_location())
                        .spawn(|_| async { std::panic::panic_any(42_u32) }, |_| {});
                    task.detach();
                    match mode.to_str().unwrap() {
                        "foreground" => {
                            dispatcher.dispatch_on_main_thread(runnable, gpui::Priority::Medium)
                        }
                        "work" => dispatcher.dispatch(runnable, gpui::Priority::Medium),
                        "timer" => {
                            dispatcher.dispatch_after(std::time::Duration::from_millis(1), runnable)
                        }
                        _ => unreachable!(),
                    }
                }));
                panic!("platform returned before running the task");
            }
            _ => unreachable!(),
        }
        return;
    }

    let run = |mode| {
        smol::block_on(
            smol::process::Command::new(std::env::current_exe().unwrap())
                .env("GPUI_NATIVE_PANIC_TEST", mode)
                .output(),
        )
        .unwrap()
    };
    let abort_status = run("baseline").status;
    assert!(!abort_status.success());
    let recovered = run("recovered");
    assert!(recovered.status.success());
    assert!(!String::from_utf8_lossy(&recovered.stderr).contains("observed "));
    let check_abort = |mode, context| {
        let child = run(mode);
        let stderr = String::from_utf8_lossy(&child.stderr);
        assert_eq!(child.status, abort_status, "{stderr}");
        assert_eq!(stderr.matches("observed ").count(), 1, "{stderr}");
        assert!(
            stderr.contains(&format!("observed {context} 42")),
            "{stderr}"
        );
    };
    check_abort("escape", "inner");
    #[cfg(target_os = "macos")]
    for mode in ["foreground", "work", "timer"] {
        check_abort(mode, "GCD task");
    }
    #[cfg(target_os = "windows")]
    for (mode, context) in [
        ("foreground", "Win32 platform window procedure"),
        ("work", "Win32 threadpool task"),
        ("timer", "Win32 threadpool timer"),
    ] {
        check_abort(mode, context);
    }
}

#[cfg(target_family = "wasm")]
fn main() {}
