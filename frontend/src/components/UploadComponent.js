import React, { useState } from 'react';
import axios from 'axios';

const UploadComponent = () => {
    const [image, setImage] = useState(null);
    const [qualityResult, setQualityResult] = useState(null);
    const [damageResult, setDamageResult] = useState(null);

    const handleFileChange = (e) => {
        setImage(e.target.files[0]);
        setQualityResult(null);
        setDamageResult(null);
    };

    const handleQualityPrediction = async () => {
        const formData = new FormData();
        formData.append("file", image);

        try {
            const response = await axios.post("http://localhost:8000/predict_quality/", formData);
            setQualityResult(response.data);
        } catch (error) {
            console.error("Error during quality prediction:", error);
        }
    };

    const handleDamagePrediction = async () => {
        const formData = new FormData();
        formData.append("file", image);

        try {
            const response = await axios.post("http://localhost:8000/predict_damage/", formData);
            setDamageResult(response.data);
        } catch (error) {
            console.error("Error during damage prediction:", error);
        }
    };

    return (
        <div>
            <h1>Vegetable Detection</h1>
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleQualityPrediction}>Predict Quality</button>
            <button onClick={handleDamagePrediction}>Predict Damage</button>
            
            {qualityResult && (
                <div>
                    <h2>Quality Prediction Result</h2>
                    <p><strong>Class:</strong> {qualityResult.class}</p>
                    <p><strong>Confidence:</strong> {qualityResult.confidence}</p>
                </div>
            )}

            {damageResult && (
                <div>
                    <h2>Damage Prediction Result</h2>
                    {damageResult.predictions.map((pred, index) => (
                        <div key={index}>
                            <p><strong>Class:</strong> {pred.class}</p>
                            <p><strong>Confidence:</strong> {pred.confidence}</p>
                            <p><strong>Bounding Box:</strong> {pred.bbox.join(', ')}</p>
                        </div>
                    ))}
                    <img src={`data:image/jpeg;base64,${damageResult.image_base64}`} alt="Detected" />
                </div>
            )}
        </div>
    );
};

export default UploadComponent;
