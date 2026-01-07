import os
from nanovllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    path = os.path.expanduser("~/huggingface/Qwen2.5-VL-3B-Instruct/")
    processor = AutoProcessor.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    prompts = inputs.input_ids.tolist()
    mm_inputs = [{
        "pixel_values": inputs.pixel_values,
        "image_grid_thw": inputs.image_grid_thw,
    }]
    outputs = llm.generate(prompts, sampling_params, mm_inputs=mm_inputs)

    for output in outputs:
        print(f"Output: {output['text']}")


if __name__ == "__main__":
    main()
