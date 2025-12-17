# Imports

from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle
import io
from utils import (
    load_data,
    transform_customer_service,
    transform_online_activity,
    transform_transaction_history,
    merge_data,
    clean_data,
    engineer_features,
    load_model_and_transformer
)


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Load pipeline once at startup
pipeline = load_model_and_transformer()

@app.route('/')     # Root/Homepage
def form():
    """
    Display upload form.
    """
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle file upload, run predictions, return results.
    """

    # Check for files
    try:
        required_files = ['demographics', 'service', 'activity', 'transaction']
        for file in required_files:
            if file not in request.files or request.files[file].filename == '':
                return f"Error: Missing {file} file. Please upload all 4 files.", 400   # 400 - Bad Request
            
        # Load data
        demo_df, service_df, activity_df, trans_df = load_data(
                request.files['demographics'],
                request.files['service'],
                request.files['activity'],
                request.files['transaction']
            )
        
        # Transform dataframes
        service_df = transform_customer_service(service_df)
        activity_df = transform_online_activity(activity_df)
        trans_df = transform_transaction_history(trans_df)

        # Merge dataframes
        data = merge_data(demo_df, service_df, activity_df, trans_df)

        # Save CustomerID's before modifying
        CustomerID = data['CustomerID'].copy()

        # Clean and engineer features
        data = clean_data(data)
        data = engineer_features(data)

        # Make predictions using loaded pipeline
        # The pipeline includes both the transformer and the model
        probabilities = pipeline.predict_proba(data)[:, 1]

        # Create results dataframe
        results = pd.DataFrame({
            'CustomerID': CustomerID,
            'Churn_Probability': probabilities
        })
        
        # Sort by highest churn probability first
        results = results.sort_values('Churn_Probability', ascending=False).reset_index(drop=True).round(2)

        # Return HTML with table + download option
        return render_template(
            'results.html',
            tables=[results.to_html(classes='table', index=False)],
            titles=['CustomerID', 'Churn_Probability'],
            results_csv=results.to_csv(index=False)
        )
      
    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/download_predictions')
def download_predictions():
    try:
        csv_data = request.args.get('data')
        output = io.BytesIO()
        output.write(csv_data.encode('utf-8'))
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='churn_predictions.csv'
        )
    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)