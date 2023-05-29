use std::process::{Output, Stdio};

use tokio::{process::{Command}, io::AsyncWriteExt};

// checks if python file syntax is ok
pub async fn run_mypy(py_file_path: &str) -> Result<Output, tokio::io::Error> {
    Command::new("mypy")
        .arg(py_file_path)
        .output()
        .await
}

pub async fn run_mypy_install_stub_types() -> Result<Output, std::io::Error> {
    let mut mypy_process = Command::new("mypy")
        .arg("--install-types")
        .stdin(Stdio::piped())
        .spawn()?;

    if let Some(ref mut stdin) = mypy_process.stdin {
        // Provide the input "y" to mypy
        stdin.write_all(b"y\n").await?;
    }

    let output = mypy_process.wait_with_output().await?;
    Ok(output)
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
