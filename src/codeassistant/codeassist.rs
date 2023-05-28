//call dependency checker, run through that
//assist with python (mypy), C#, unity, whatever
//call code iterator on the code

use crate::{ai::{aimanager::get_prompt_by_model, self}, filemanager::fileloader::{create_file_python, create_file_csharp, create_file_rust, create_file_cplusplus}};

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
                create_file_python(&gpt_response).await?;
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