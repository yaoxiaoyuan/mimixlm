### Convert Llama3.2

- **This guide demonstrates how to Adapting Llama3.2-1B-Instruct model for our framework.**

- **Note: All commands below must be executed from the project root directory.(i.e., the directory containing the mimixlm.py)**

- **Download the Llama3.2-1B model. (Requires the Hugging Face Transformers library)**
  
  - sh examples/convert_llama3/download_llama3.2-1b.sh  
  
- **Convert the model.**
  
  - Prepare 'model_config.json' and 'tokenizer.model' file (I have already prepared) .
  - sh examples/convert_llama3/convert.sh
  
- **Testing Conversion Results**

  - **Run:** python examples\convert_llama3\convert_hf_weights.py --hf_weight_path model\Llama-3.2-1B-Instruct --convert_path model\converted_llama3_1B_instruct

  - ```
    User:
    why sky looks blue?
    Assistant:
    The sky appears blue because of a phenomenon called Rayleigh scattering, named after the British physicist Lord Rayleigh, who first described it in the late 19th century.
    
    Here's what happens:
    
    1. **Sunlight enters Earth's atmosphere**: When the sun's light enters our atmosphere, it encounters tiny molecules of gases such as nitrogen (N2) and oxygen (O2).
    2. **Light is scattered**: These molecules scatter the light in all directions, but they scatter shorter (blue) wavelengths more than longer (red) wavelengths. This is known as Rayleigh scattering.
    3. **Blue light is scattered more**: As a result, the blue light is dispersed in all directions, reaching our eyes from every part of the sky. This is why the sky typically appears blue during the daytime.
    4. **Red light is not scattered as much**: The longer wavelengths of light, such as red, are not scattered as much, so they continue to travel in a more direct path to our eyes.
    
    This is why the sky appears blue, especially during the daytime when the sun is overhead. At sunrise and sunset, the light has to travel through more of the Earth's atmosphere, which scatters the shorter wavelengths, making the sky appear more red or orange.
    
    It's worth noting that the color of the sky can also be affected by other factors, such as:
    
    * **Dust and pollution**: Tiny particles in the air can scatter light and give the sky a hazy or gray appearance.
    * **Clouds**: Clouds can reflect and scatter light, making the sky appear white or gray.
    * **Time of day**: During sunrise and sunset, the light has to travel through more of the atmosphere, which scatters the shorter wavelengths, making the sky appear more red or orange.
    
    I hope that helps you understand why the sky looks blue!
    ```

    
