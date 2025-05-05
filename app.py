import gradio as gr
from prediction import get_prediction_results, list_available_records

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
        
        def display_results(ecg_data, sample_id):
            output = get_prediction_results(ecg_data, sample_id)
            
            return f"""
            ### Prediction Results
            - **Prediction**: {output['classification']}
            - **Confidence**: {output['confidence']}
            """

        # Make sure to specify the outputs parameter
        predict_btn.click(
            display_results,
            inputs=[ecg_data, sample_id],
            outputs=result
        )



if __name__ == "__main__":
    demo.launch()




