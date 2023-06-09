use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::{PathBuf, Path};

#[derive(Debug, Serialize)]
struct AIDiffuserRequest {
    prompt: String,
    height: i32,
    width: i32,
    num_inference_steps: i32,
    guidance_scale: f32,
    use_columns: bool,
}

#[derive(Debug, Deserialize)]
struct AIDiffuserResponse {
    image_path: String,
}
pub async fn generate_stable_diffuse_image(
    host_address: &str,
    prompt: &str,
    height: i32,
    width: i32,
    num_inference_steps: i32,
    guidance_scale: f32,
    img_count: i32,
    use_columns: bool,
) -> Result<(bool, String), Box<dyn Error>> {
    let url = format!(
        "http://{}/generateimg?prompt={}&height={}&width={}&num_inference_steps={}&guidance_scale={}&img_count={}&use_columns={}",
        host_address, prompt, height, width, num_inference_steps, guidance_scale, img_count, use_columns
    );
    println!("Requesting: {}", url);
    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Failed to send request: {}", e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Request to {} failed with status code: {}",
            url,
            response.status()
        )
        .into());
    }

    let response_text = response
        .text()
        .await
        .map_err(|e| format!("Failed to retrieve response body: {}", e))?;

    let json: Value = serde_json::from_str(&response_text)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;

    let image_path = json["image_path"]
        .as_str()
        .ok_or("image_path not found or not a string")?;

    let filtered_path = filter_path(image_path)
        .ok_or_else(|| format!("Failed to filter the image path: {}", image_path))?;

    Ok((true, filtered_path))
}




fn get_image_path(json_str: &str) -> Option<String> {
    // Parse the JSON string
    let parsed: Result<Value, _> = serde_json::from_str(json_str);
    match parsed {
        Ok(Value::Object(obj)) => {
            // Check if the object contains an "image_path" key
            match obj.get("image_path") {
                Some(Value::String(path)) => Some(path.to_string()),
                _ => None,
            }
        }
        _ => None,
    }
}

fn filter_path(linux_path: &str) -> Option<String> {
    if let Some(index) = linux_path.rfind('/') {
        if let Some(filename) = linux_path.get(index + 1..) {
            return Some(format!("./artificialintelligence/gen_pics/{}", filename));
        }
    }
    None
}


fn convert_to_windows_path(linux_path: &str) -> String {
    let path = Path::new(linux_path);
    let mut buf = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::RootDir => buf.push("\\"),
            std::path::Component::Normal(os_str) => {
                if let Some(s) = os_str.to_str() {
                    buf.push(s);
                }
            }
            _ => (),
        }
    }
    buf.to_string_lossy().to_string()
}