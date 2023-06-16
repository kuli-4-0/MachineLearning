# Event Recomenders with Deep & Cross Network (DCN) Algorithm, Tensorflow Recomenders (TFRS)


Deep & Cross Network (DCN) is an algorithm used in the field of predictive modeling, particularly in recommendation problems. This algorithm combines the power of Deep Learning models and Linear models, enabling it to model complex relationships between input features.

## How to Run the Flask File

1. Make sure all the following requirements are installed in your environment:

    ```
    Flask==2.3.2
    ipykernel==6.20.2
    keras==2.11.0
    numpy==1.24.1
    pandas==1.5.3
    scikit-learn==1.2.1
    tensorboard==2.11.2
    tensorboard-data-server==0.6.1
    tensorboard-plugin-wit==1.8.1
    tensorflow==2.11.0
    tensorflow-recommenders==0.7.3
    requests==2.28.2
    flask_cors==3.0.10
    ```

2. Download or copy the Flask file containing the implementation of DCN into your directory.

3. Open a terminal or command prompt and navigate to the directory containing the Flask file.

4. Activate the virtual environment (optional but recommended) by running the following command:

    ```
    source <virtual_environment_name>/bin/activate
    ```

    or if you're using anaconda env, you can activate the virtual environment (optional but recommended) by running the following command:

    ```
    conda activate <virtual_environment_name>
    ```

5. Run the following command to start the Flask application:

    ```
    flask run
    ```

   If all the above steps are successful, you will see an output indicating that the Flask application is running and can be accessed through a specific URL.

6. Open a web browser and access the URL displayed in the terminal to access the Flask application.

## Accessing the API

You can access the API using the following code snippet:
```Node Js/Javascript
const axios = require('axios');

const data = {
  "id_user": "2",
  "genre": "pop"
};

axios.post('https://myservice-brcxr6bfxq-uc.a.run.app/predict', data)
  .then(response => {
    console.log(response.data);
  })
  .catch(error => {
    console.error(error);
  });

```
The data object in the code represents the payload to be sent to the API. You can adjust the data according to your needs.

Please make sure you have installed the required dependencies, including Axios, by running npm install axios before running the code above.


## Requirements

The following is a list of requirements needed to run the Flask application using the Deep & Cross Network (DCN) algorithm:

Flask==2.3.2
ipykernel==6.20.2
keras==2.11.0
numpy==1.24.1
pandas==1.5.3
scikit-learn==1.2.1
tensorboard==2.11.2
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow==2.11.0
tensorflow-recommenders==0.7.3
requests==2.28.2
flask_cors==3.0.10

You can install all the requirements above using pip by running the following command:

pip install -r requirements.txt



