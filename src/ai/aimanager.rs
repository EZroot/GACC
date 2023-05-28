use reqwest::Client;

use super::models::{gpt::chat_gpt, davinci::chat_davinci};

pub enum AIModelType {
    GptModel,
    DavinciModel,
}

static mut API_KEY: Option<String> = None;

pub fn set_key(api_key: &str) {
    println!("Api key set: {}", api_key);
    unsafe {
        API_KEY = Some(api_key.to_string());
    }
}

fn get_api_key() -> Option<String> {
    unsafe { API_KEY.clone() }
}


pub async fn get_prompt_by_model(
    prompt: String,
    model_type: AIModelType,
) -> Result<(bool, String), reqwest::Error> {
    let key = get_api_key().unwrap();

    //req client
    let client = Client::new();

    let mut model_response = (false, String::new());

    match model_type {
        AIModelType::GptModel => {
            println!("Getting response from GPT model");
            let system_content: String = "You are a programmer assistant.".to_string();
            model_response = chat_gpt(&client, &key, system_content, prompt).await.unwrap();
        }
        AIModelType::DavinciModel => {
            println!("Getting response from Davinci model");
            model_response = chat_davinci(&client, &key , prompt).await.unwrap();
        }
    }
    println!("Model request finished.");
    Ok(model_response)
}
