use reqwest::Error;
use urlencoding::encode;

pub struct AIRequestor {}

impl AIRequestor {
    pub async fn send_ai_prompt_request(prompt : &str, image_count : i32, width : i32, height : i32, inpaint_image_filepath:&str) -> Result<String, Error> {
        let prompt_encoded_string = encode(prompt);
        let filepath_encoded_string = encode(inpaint_image_filepath);
        let url = format!("http://localhost:6969//lineartanime?prompt={}&img_count={}&width={}&height={}&filepath={}", prompt_encoded_string, image_count,width,height, filepath_encoded_string);

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
