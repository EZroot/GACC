AI Support
- Stable Diffusion
- Controlnets
- Code Generation

Pico Picasso - Image editor + AI Gen
* Draw and generate
* Regenerate images
* Edit images and regenerate regions
![Pico Picasso](https://cdn.discordapp.com/attachments/1051535103223930984/1118021969200296016/image.png)
![Pico Picasso](https://cdn.discordapp.com/attachments/1112415182975676497/1118063562926071808/image.png)

Stable Diffusion
curl -X GET 'http://127.0.0.1:6969/stablediffusion?prompt=a%20red%20gorilla&height=768&width=768&num_inference_steps=50&img_count=1&use_columns=True&negative_prompt=cropped%2C%20lowres%2C%20poorly%20drawn%20face%2C%20out%20of%20frame%2C%20poorly%20drawn%20hands%2C%20blurry%2C%20bad%20art%2C%20blurred%2C%20text%2C%20watermark%2C%20disfigured%2C%20deformed%2C%20closed%20eyes'


curl -X GET 'http://localhost:port/openpose' \
--data-urlencode 'prompt=a%20photo%20of%20a%20banana%20on%20a%20pyramid' \
--data-urlencode 'negative_prompt=cropped%2C%20lowres%2C%20poorly%20drawn%20face%2C%20out%20of%20frame%2C%20poorly%20drawn%20hands%2C%20blurry%2C%20bad%20art%2C%20blurred%2C%20text%2C%20watermark%2C%20disfigured%2C%20deformed%2C%20closed%20eyes' \
--data-urlencode 'filepath=derp.png' \
--data-urlencode 'num_inference_steps=50' \
--data-urlencode 'img_count=1' \
--data-urlencode 'use_columns=true'

