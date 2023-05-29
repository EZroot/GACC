//call dependency checker, run through that
//assist with python (mypy), C#, unity, whatever
//call code iterator on the code
use crate::{
    ai::{self, aimanager::get_prompt_by_model},
    codeassistant::dependencies::dependants::python::{
        mypy::run_mypy_install_stub_types, pip::run_pip_install,conda::run_conda_install
    },
    filemanager::fileloader::{
        create_file_cplusplus, create_file_csharp, create_file_python, create_file_rust,
    }, clicommands::commandparser::restart_with_args,
};
use colored::*;
use regex::Regex;

use super::dependencies::dependencychecker::{
    check_codeassistant_dependencies, DependencyType, SyntaxCheckStatus,
};

static mut USE_CONDA_OVER_PIP: Option<bool> = None;
static mut CODE_ITERATIONS: Option<i32> = None;

pub fn set_code_iterations(code_iterations:i32)
{
    println!("set code interations: {}", code_iterations);
    unsafe{
        CODE_ITERATIONS = Some(code_iterations);
    }
}

fn get_code_iterations() -> Option<i32> {
    unsafe { CODE_ITERATIONS.clone() }
}

pub fn set_conda_over_pip(use_conda_over_pip: bool) {
    println!("use_conda_over_pip set: {}", use_conda_over_pip);
    unsafe {
        USE_CONDA_OVER_PIP = Some(use_conda_over_pip);
    }
}

fn get_conda_over_pip() -> Option<bool> {
    unsafe { USE_CONDA_OVER_PIP.clone() }
}

pub enum CodeType {
    Python,
    CSharp,
    CPlusPlus,
    Rust,
}

pub async fn assist_create_code(
    file_name: &str,
    code_suggestion: &str,
    code_type: CodeType,
) -> Result<(), Box<dyn std::error::Error>> {
    match code_type {
        CodeType::Python => {
            let formatted_request = format!("Please remain quiet and only provide a python script for the following prompt. {}", code_suggestion);
            let formatted_request_clone = formatted_request.clone();
            let request_bytes = formatted_request_clone.as_bytes();

            let (success_status, gpt_response) =
                get_prompt_by_model(formatted_request, ai::aimanager::AIModelType::DavinciModel)
                    .await?;
            if success_status {
                let file_path = create_file_python(file_name, &gpt_response).await?;
                let python_code_iterations = get_code_iterations().unwrap();
                let mut iteration_counter = 0;
                'mainloop: loop {
                    if iteration_counter > python_code_iterations {
                        println!("Iteration limit reached!\nFAILED to fix {}\n Breaking loop...", file_name.red());
                        break;
                    }
                    let iteration_info = format!("{}/{}", iteration_counter, python_code_iterations);
                    println!("Looping interation {}", iteration_info.bright_magenta());
                    iteration_counter+=1;
                    let status =
                        check_codeassistant_dependencies(&file_path, DependencyType::Python).await;
                    match status {
                        SyntaxCheckStatus::Success => {
                            let success = "Python syntax passed successfully";
                            println!("Congrats! {}", success.green());
                            println!("Breaking loop...");
                            break;
                        }
                        SyntaxCheckStatus::Failed(failed_str) => {
                            //We have to check the imports and pip install them
                            let re = Regex::new(r#""([^"]+)""#).unwrap();
                            let error_count = failed_str.lines().filter(|line| line.contains("error:")).count();
                            for lines in failed_str.lines() {
                                println!("Analyzing line: {}", lines.purple());
                                for capture in re.captures_iter(&lines) {
                                    let module_name =
                                        capture.get(0).unwrap().as_str().replace("\"", "");
                                        //More Library stubs
                                    if lines.contains("Skipping analyzing") {
                                        println!("[Warning] Skipping error: {}", &lines.cyan());
                                        if error_count <= 1 {
                                            let success = "SUCCESS";
                                            println!("Error count is 1 or less, so were assuming: {}",success.green());
                                            println!("Breaking main loop...");
                                            break 'mainloop;
                                        }
                                    } else if lines.contains("missing library stubs") {
                                        println!("Missing library stubs: {}", module_name.red());
                                        let stub_success = run_mypy_install_stub_types().await?;
                                        if (stub_success.status.success()) {
                                            let success = "Success";
                                            println!("Stubs installed: {}", success.green());
                                        } else {
                                            let success = "Error";
                                            println!("Stubs installed: {}", success.red());
                                        }
                                        //Library Stubs
                                    } else if lines.contains("syntax error") {
                                        println!("Syntax Error. Restarting program with same prompt...");
                                        let new_string = String::from_utf8_lossy(request_bytes).to_string();
                                        let custom_args = vec!["-npy".to_string(), file_name.to_string(), new_string];
                                        restart_with_args(&custom_args).await;
                                    }else if lines.contains("Library stubs not installed") {
                                        println!(
                                            "Stubs not installed for module: {} \n Attempting to install...",
                                            module_name.red()
                                        );
                                        let stub_success = run_mypy_install_stub_types().await?;
                                        if stub_success.status.success() {
                                            let success = "Success";
                                            println!("Stubs installed: {}", success.green());
                                        } else {
                                            let success = "Error";
                                            println!("Stubs installed: {}", success.red());
                                        }
                                    } else {
                                        //Module Imports
                                        println!("Module import needed: {}", module_name.red());
                                        //println!("Module name: {}", module_name.green());
                                        let use_conda = get_conda_over_pip().unwrap();
                                        if use_conda {
                                            let status = run_conda_install(&module_name).await;
                                            match status {
                                                Ok(output) => {
                                                    if output.status.success() {
                                                        println!(
                                                            "CONDA Successfully installed module {}",
                                                            module_name.green()
                                                        );
                                                        println!("Breaking line loop, moving to next line");
                                                        break;
                                                    } else {
                                                        let output_str =
                                                            String::from_utf8_lossy(&output.stdout);
                                                        println!(
                                                        "CONDA Failed to install module {} \nError\n{}\n",
                                                        module_name.red(),
                                                        output_str.red()
                                                    );
                                                    }
                                                }
                                                Err(err) => {
                                                    println!("CONDA ERROR: {}", err.to_string().red());
                                                    break 'mainloop;
                                                }
                                            }
                                        }else{
                                        let status = run_pip_install(&module_name).await;
                                        match status {
                                            Ok(output) => {
                                                if output.status.success() {
                                                    println!(
                                                        "PIP Successfully installed module {}",
                                                        module_name.green()
                                                    );
                                                    println!("Breaking line loop, moving to next line");
                                                    break;
                                                } else {
                                                    let output_str =
                                                        String::from_utf8_lossy(&output.stdout);
                                                    println!(
                                                    "PIP Failed to install module {} \nError\n{}\n",
                                                    module_name.red(),
                                                    output_str.red()
                                                );
                                                }
                                            }
                                            Err(err) => {
                                                println!("PIP ERROR: {}", err.to_string().red());
                                                break 'mainloop;
                                            }
                                        }
                                    }
                                    }
                                }
                            }
                            println!("Python syntax FAILED\n {}", failed_str.red())
                        }
                        SyntaxCheckStatus::Error => {
                            println!("check code assist unknown error");
                            break;
                        }
                    }
                }
            }
        }
        CodeType::CSharp => {
            let formatted_request = format!("{} in C#.", code_suggestion);
            let (success_status, gpt_response) =
                get_prompt_by_model(formatted_request, ai::aimanager::AIModelType::DavinciModel)
                    .await?;
            if success_status {
                create_file_csharp(file_name,&gpt_response).await?;
            }
        }
        CodeType::CPlusPlus => {
            let formatted_request = format!("{} in C++.", code_suggestion);
            let (success_status, gpt_response) =
                get_prompt_by_model(formatted_request, ai::aimanager::AIModelType::DavinciModel)
                    .await?;
            if success_status {
                create_file_cplusplus(file_name,&gpt_response).await?;
            }
        }
        CodeType::Rust => {
            let formatted_request = format!("{} in Rust.", code_suggestion);
            let (success_status, gpt_response) =
                get_prompt_by_model(formatted_request, ai::aimanager::AIModelType::DavinciModel)
                    .await?;
            if success_status {
                create_file_rust(file_name,&gpt_response).await?;
            }
        }
    }

    Ok(())
}
