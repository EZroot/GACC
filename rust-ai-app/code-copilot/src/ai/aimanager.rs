use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::models::{gpt::chat_gpt, davinci::chat_davinci, stablediffusion::generate_stable_diffuse_image};

pub enum AIModelType {
    GptModel,
    DavinciModel,
    StableDiffuseModel,
}

#[derive(Debug,Clone, Serialize, Deserialize)]
pub struct StableDiffusionConfig {
    pub height : i32,
    pub width : i32,
    pub num_inference_steps : i32,
    pub guidance_scale : f32,
    pub img_count : i32,
    pub use_columns : bool,
}

static mut API_KEY: Option<String> = None;
static mut STABLE_DIFFUSE_HOST_ADDRESS: Option<String> = None;
static mut STABLE_DIFFUSE_CONFIG: Option<StableDiffusionConfig> = None;

pub fn set_stable_diffuse_config(stable_diffusion_config : StableDiffusionConfig) {
    unsafe {
        STABLE_DIFFUSE_CONFIG = Some(stable_diffusion_config);
    }
}

fn get_stable_diffuse_config() -> Option<StableDiffusionConfig> {
    unsafe { STABLE_DIFFUSE_CONFIG.clone() }
}

pub fn set_stable_diffuse_host_address(host_address: &str) {
    println!("Stable Diffuse Address: {}", host_address);
    unsafe {
        STABLE_DIFFUSE_HOST_ADDRESS = Some(host_address.to_string());
    }
}

fn get_stable_diffuse_host_address() -> Option<String> {
    unsafe { STABLE_DIFFUSE_HOST_ADDRESS.clone() }
}


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
    let diffuse_host_address = get_stable_diffuse_host_address().unwrap();
    let diffuse_config = get_stable_diffuse_config().unwrap();
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
        AIModelType::StableDiffuseModel => {
            let host_address = diffuse_host_address;
            let height = diffuse_config.height;
            let width = diffuse_config.width;
            let num_inference_steps = diffuse_config.num_inference_steps;
            let guidance_scale = diffuse_config.guidance_scale;
            let img_count = diffuse_config.img_count;
            let use_columns = diffuse_config.use_columns;
            println!("Getting response from StableDiffusion model");
            model_response = generate_stable_diffuse_image(
                &host_address,
                &prompt,
                height,
                width,
                num_inference_steps,
                guidance_scale,
                img_count,
                use_columns,
            )
            .await.unwrap();
        }
    }
    println!("Model request finished. {} : {}",model_response.0, model_response.1);
    Ok(model_response)
}
