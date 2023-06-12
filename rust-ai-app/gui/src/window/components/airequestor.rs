use reqwest::Error;
use urlencoding::encode;

pub struct AIRequestor {}

impl AIRequestor {
    pub async fn send_ai_prompt_request(prompt : &str, image_count : i32, width : i32, height : i32) -> Result<String, Error> {
        let encoded_string = encode(prompt);
        let url = format!("http://localhost:6969/stablediffusion?prompt={}&img_count={}&width={}&height={}", &encoded_string, image_count,width,height);

        let response = reqwest::get(&url).await?;

        if response.status().is_success() {
            let body = response.text().await?;
            println!("Response text: {}", body);
            Ok(body)
        } else {
            println!("Unexpected server response: {}", response.status());
            Ok("Fail to get server response".to_string())
        }    
    }
}
