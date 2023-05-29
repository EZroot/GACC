use std::process::Output;
use tokio::process::{Command};

pub async fn run_conda_install(module_name: &str) -> Result<Output, tokio::io::Error> {
    println!("Conda Command: conda install {}",&module_name);
    Command::new("conda")
       .arg("install")
       .arg(module_name)
       .arg("-y")
       .output()
       .await
}

pub async fn check_conda_installed() -> bool {
    let output = Command::new("conda").arg("--version").output().await;

    match output {
        Ok(_) => true,  // pip is installed
        Err(_) => false,  // pip is not installed
    }
}

pub async fn install_python_and_conda() -> Result<std::process::Output, tokio::io::Error> {
    Command::new("apt")
        .arg("install")
        .arg("-y")
        .arg("python3") // Update with the appropriate package name for Python 3 on your Linux distribution
        .output()
        .await
}