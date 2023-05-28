//call dependency checker, run through that
//assist with python (mypy), C#, unity, whatever
//call code iterator on the code
use colored::*;
use regex::Regex;
use crate::{
    ai::{aimanager::get_prompt_by_model, self}, 
    filemanager::fileloader::{create_file_python, create_file_csharp, create_file_rust, create_file_cplusplus}, codeassistant::dependencies::dependants::python::pip::run_pip_install,
};

use super::dependencies::dependencychecker::{check_codeassistant_dependencies, DependencyType, SyntaxCheckStatus};

pub enum CodeType {
    Python,
    CSharp,
    CPlusPlus,
    Rust,
}

pub async fn assist_create_code(code_suggestion:&str, code_type:CodeType) -> Result<(), Box<dyn std::error::Error>>{
    match code_type {
        CodeType::Python => {
            let formatted_request = format!("{} in Python.", code_suggestion);
            let (success_status, gpt_response) = get_prompt_by_model(formatted_request, ai::aimanager::AIModelType::DavinciModel).await?;
            if success_status {
                let file_path = create_file_python(&gpt_response).await?;

                loop {
                    println!("Looping interation...");
                    let status = check_codeassistant_dependencies(&file_path, DependencyType::Python).await;
                    match status {
                        SyntaxCheckStatus::Success => { 
                            let success = "Python syntax passed successfully";
                            println!("Congrats! {}", success.green());
                            println!("Breaking loop...");
                            break;
                        },
                        SyntaxCheckStatus::Failed(failed_str) => {
                            //We have to check the imports and pip install them
                            let re = Regex::new(r#""([^"]+)""#).unwrap();
                            for capture in re.captures_iter(&failed_str) {
                                let module_name = capture.get(0).unwrap().as_str().replace("\"", "");
                                println!("Module name: {}", module_name.green());
                                let status = run_pip_install(&module_name).await;
                                match status {
                                    Ok(output) => {
                                        if output.status.success() {
                                            println!("PIP Successfully installed module {}", module_name.green());
                                            println!("Breaking loop...");
                                            break;
                                        }else{
                                            let output_str = String::from_utf8_lossy(&output.stdout);
                                            println!("PIP Failed to install module {} \nError\n{}\n", module_name.red(), output_str.red());
                                        }
                                    },
                                    Err(err) => {
                                        println!("PIP ERROR: {}",err.to_string().red());
                                        break;
                                    },
                                }
                            }
                            println!("Python syntax FAILED\n {}", failed_str.red())
                        },
                        SyntaxCheckStatus::Error => { println!("check code assist unknown error"); 
                        break;},
                    }
                }
            }        
        }
        CodeType::CSharp => {
            let formatted_request = format!("{} in C#.", code_suggestion);
            let (success_status, gpt_response) = get_prompt_by_model(formatted_request, ai::aimanager::AIModelType::DavinciModel).await?;
            if success_status {
                create_file_csharp(&gpt_response).await?;
            }        
        }
        CodeType::CPlusPlus => {
            let formatted_request = format!("{} in C++.", code_suggestion);
            let (success_status, gpt_response) = get_prompt_by_model(formatted_request, ai::aimanager::AIModelType::DavinciModel).await?;
            if success_status {
                create_file_cplusplus(&gpt_response).await?;
            }        
        }
        CodeType::Rust => {
            let formatted_request = format!("{} in Rust.", code_suggestion);
            let (success_status, gpt_response) = get_prompt_by_model(formatted_request, ai::aimanager::AIModelType::DavinciModel).await?;
            if success_status {
                create_file_rust(&gpt_response).await?;
            }        
        }

    }


    Ok(())
}