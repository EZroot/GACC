use reqwest::Client;

use super::models::{gpt::chat_gpt, davinci::chat_davinci};

pub enum AIModelType {
    GptModel,
    DavinciModel,
}

static mut API_KEY: Option<String> = None;

pub fn set_key(api_key: &str) {
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
            let system_content: String = "You are a programmer assistant.".to_string();
            model_response = chat_gpt(&client, &key, system_content, prompt).await.unwrap();
        }
        AIModelType::DavinciModel => {
            let system_content: String = "A table summarizing the fruits from Goocrux:\n\nThere are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy. There are also loheckles, which are a grayish blue fruit and are very tart, a little bit like a lemon. Pounits are a bright green color and are more savory than sweet. There are also plenty of loopnovas which are a neon pink flavor and taste like cotton candy. Finally, there are fruits called glowls, which have a very sour and bitter taste which is acidic and caustic, and a pale orange tinge to them.\n\n| Fruit | Color | Flavor |".to_string();

            model_response = chat_davinci(&client, &key , prompt).await.unwrap();
        }
    }

    Ok(model_response)
}
