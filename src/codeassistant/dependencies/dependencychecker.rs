use std::io::{self, Write};
use colored::*;
use regex::Regex;

use crate::codeassistant::dependencies::dependants::python::pip::install_python_and_pip;

use super::dependants::python::{mypy::{install_mypy, check_mypy_installed, run_mypy}, pip::check_pip_installed};

//for all dependants, check if available, download if not
pub enum DependencyType {
    Python,
    CSharp,
    CPlusPlus,
    Rust,
}

pub enum SyntaxCheckStatus {
    Success,
    Failed(String),
    Error,
}

pub async fn check_codeassistant_dependencies(file_path:&str, dependency: DependencyType) -> SyntaxCheckStatus {
    match dependency {
        DependencyType::Python => {
            return check_dependencies_python(file_path).await;
        },
        DependencyType::CSharp => {
            return SyntaxCheckStatus::Error;
        },
        DependencyType::CPlusPlus => {
            return SyntaxCheckStatus::Error;
        },
        DependencyType::Rust => {
            return SyntaxCheckStatus::Error;
        },
    }
    
}

async fn check_dependencies_python(file_path:&str) -> SyntaxCheckStatus {
    println!("Checking python and pip is installed...");
    check_python_dependency().await; //make sure python is installed (so we have pip)
    println!("Checking mypy is installed...");
    return check_mypy_dependency(file_path).await; //check file syntax, if mypy isnt installed, install it with pip
}

async fn check_python_dependency() -> SyntaxCheckStatus {
    if !check_pip_installed().await {
        print!("Python3 is not installed. Do you want to install it? (y/n): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let choice = input.trim().to_lowercase();

        if choice.contains('y') {
            match install_python_and_pip().await {
                Ok(_) => { 
                    println!("Python3 & pip installed successfully.");
            },
                Err(err) => eprintln!("Failed to install Python3: {}", err),
            }
        } else {
            println!("Python3 installation skipped.");
        }
    } else {
        println!("Python3 is already installed.");
    }
    return SyntaxCheckStatus::Error;
}

async fn check_mypy_dependency(file_path:&str) -> SyntaxCheckStatus {
    if !check_mypy_installed().await {
        print!("mypy is not installed. Do you want to install it? (y/n): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let choice = input.trim().to_lowercase();

        if choice.contains('y') {
            match install_mypy().await {
                Ok(_) => { 
                    println!("mypy installed successfully.");
                    let status = run_mypy(file_path).await;
                    match status {
                        Ok(output) => {
                            // Check the status code of the output
                            if output.status.success() {
                                return SyntaxCheckStatus::Success;
                            } else {
                                let output_str = String::from_utf8_lossy(&output.stdout);
                                return SyntaxCheckStatus::Failed(output_str.to_string());
                            }
                        }
                        Err(error) => {
                            return SyntaxCheckStatus::Error;
                        }
                    } 
            },
                Err(err) => eprintln!("Failed to install mypy: {}", err),
            }
        } else {
            println!("mypy installation skipped.");
        }
    } else {
        println!("mypy is already installed.");
        let status = run_mypy(file_path).await;
        match status {
            Ok(output) => {
                // Check the status code of the output
                if output.status.success() {
                    return SyntaxCheckStatus::Success;
                } else {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    return SyntaxCheckStatus::Failed(output_str.to_string());
                }
            }
            Err(error) => {
                println!("Error: Likely a syntax error in mypy{:?}", error);
                return SyntaxCheckStatus::Error;
            }
        } 
    }
    return SyntaxCheckStatus::Error;
}