use std::process::Output;

use tokio::process::{Command};

// checks if python file syntax is ok
pub async fn run_mypy(py_file_path: &str) -> Result<Output, tokio::io::Error> {
    Command::new("mypy")
        .arg(py_file_path)
        .output()
        .await
}

pub async fn check_mypy_installed() -> bool {
    let output = Command::new("mypy").arg("--version").output().await;

    match output {
        Ok(_) => true,  // mypy is installed
        Err(_) => false,  // mypy is not installed
    }
}

pub async fn install_mypy() -> Result<std::process::Output, tokio::io::Error> {
    Command::new("pip")
        .arg("install")
        .arg("mypy")
        .output()
        .await
}
