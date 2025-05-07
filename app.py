import gradio as gr
from prediction import list_available_records, read_ecg_signal, get_deepfake_ecg
import requests

def update_sample_visibility(choice):
    return gr.update(visible=(choice == "MIT database"))

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # ECG Detection Application
        This is a demo of fake ECG detection using a pre-trained model.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Example Predictions")

            ecg_data = gr.Dropdown(
                label="Select an example ECG signal",
                choices=[
                    "MIT database",
                    "Deepfake sample"
                ],
                value="MIT database",
                interactive=True,
            )
            
            # Always create the sample_id component, but control its visibility
            sample_id = gr.Dropdown(
                label="Select a sample ID",
                choices=list_available_records(),
                value=list_available_records()[0] if list_available_records() else None,
                interactive=True,
                visible=True,
            )

            sample_rate = gr.Number(
                label="Sample Rate (Hz)",
                precision=0,
                value=500,
                interactive=True,
                visible=True
            )
            
            predict_btn = gr.Button(
                "Get Prediction",
                variant="primary",
                interactive=True,
            )


        with gr.Column():
            gr.Markdown("## Prediction Results")
            result = gr.Markdown(
                """
                Results will be displayed here after you click the button.
                """
            )
        
        # Make sample_id dropdown visible only when MIT database is selected
            
        ecg_data.change(fn=update_sample_visibility, inputs=ecg_data, outputs=sample_id)
        
        def display_results(ecg_data, sample_id, sample_rate):
            output = send_request(ecg_data, sample_id, sample_rate)
            
            return f"""
            ### Prediction Results
            - **Prediction**: {output['classification']}
            - **Confidence**: {output['confidence']}
            """

        # Make sure to specify the outputs parameter
        predict_btn.click(
            display_results,
            inputs=[ecg_data, sample_id, sample_rate],
            outputs=result
        )


def send_request(data_name, sample_id, sample_rate):
    if data_name == "MIT database":
        ecg_signal, fields = read_ecg_signal(sample_id)  # Example record ID
        ecg_signal = ecg_signal[:, 0]  # Use the first channel if multi-channel
        # sample_rate = fields['fs']
    else:
        ecg_signal = get_deepfake_ecg()[:,1]  # Example generated signal
        # sample_rate = 500
    
    # write the signal to a  txt file
    with open("user_files/ecg_signal.txt", "w") as f:
        for value in ecg_signal:
            f.write(f"{value}\n")
    
    result = requests.post(
        "http://localhost:5000/api/predict",
        files={"ecg_file": open("user_files/ecg_signal.txt", "rb")},
        data={"sample_rate": sample_rate}
    ).json()



    # 2. Classify the signal
    # print(f"Classification: {result['classification']}")
    # print(f"Confidence: {result['confidence']:.2f}")
    return result

if __name__ == "__main__":
    demo.launch()




