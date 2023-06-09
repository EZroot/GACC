use serde::{Serialize, Deserialize};
use reqwest::Response as ReqwestResponse;

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize)]
struct RequestBody {
    model: String,
    prompt: String,
    temperature: f32,
    max_tokens: usize,
    top_p: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
}

#[derive(Serialize, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Serialize, Deserialize)]
struct Choice {
    text: String,
}

pub async fn chat_davinci(client: &reqwest::Client, api_key: &str, prompt: String) -> Result<(bool, String), reqwest::Error> {
    let prompt = RequestBody {
        model: String::from("text-davinci-003"),
        prompt,
        temperature: 0.0,
        max_tokens: 3000,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
    };

    let response = client.post("https://api.openai.com/v1/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&prompt).unwrap())
        .send()
        .await?;

    // Check if there are any errors during the request
    if response.status().is_success() {
        let response_body = response.text().await?;
        let res: ChatResponse = serde_json::from_str(&response_body).unwrap();

        if let Some(choice) = res.choices.first() {
            let reply = choice.text.clone();

            // Return success status as true and the reply
            return Ok((true, reply));
        }

        // Return success status as true and the reply
        Ok((false, "API Valid, but no response".to_string()))
    } else {
        // Return success status as false and an empty reply
        Ok((false, "No API key or wrong body request set.".to_string()))
    }
}
