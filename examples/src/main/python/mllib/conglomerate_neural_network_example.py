#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Conglomerate Neural Network Ensemble Example

This example demonstrates a jury-based neural network ensemble method where
multiple MultilayerPerceptronClassifiers work together to make predictions
through voting/averaging, similar to a Random Forest but for neural networks.

The "conglomerate" approach combines predictions from multiple neural network
architectures (jury members) to improve robustness and accuracy.

Usage:
    spark-submit conglomerate_neural_network_example.py

This ensemble method is particularly useful for mobile broadband neuro network
processing where distributed consensus provides better predictions.
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, count, expr
import numpy as np


class ConglomerateNeuralNetwork:
    """
    A jury-based ensemble of neural networks that combines predictions
    from multiple MultilayerPerceptronClassifier models.
    
    This implements a conglomerate approach where each neural network
    acts as a "jury member" contributing to the final decision.
    """
    
    def __init__(self, num_models=3, max_iter=100, seed=1234):
        """
        Initialize the conglomerate neural network ensemble.
        
        Args:
            num_models: Number of neural network models in the jury
            max_iter: Maximum iterations for each model
            seed: Random seed for reproducibility
        """
        self.num_models = num_models
        self.max_iter = max_iter
        self.seed = seed
        self.models = []
        self.architectures = []
        
    def _create_diverse_architectures(self, input_size, output_size):
        """
        Create diverse neural network architectures for the ensemble.
        Each architecture varies in hidden layer configuration.
        
        Args:
            input_size: Number of input features
            output_size: Number of output classes
            
        Returns:
            List of layer configurations
        """
        architectures = []
        
        # Architecture 1: Single hidden layer with moderate neurons
        architectures.append([input_size, 5, output_size])
        
        # Architecture 2: Two hidden layers
        architectures.append([input_size, 6, 4, output_size])
        
        # Architecture 3: Deeper network with smaller layers
        architectures.append([input_size, 4, 4, 4, output_size])
        
        # Use only the number of models requested
        return architectures[:self.num_models]
    
    def fit(self, train_data, input_size=4, output_size=3):
        """
        Train all neural network models in the conglomerate ensemble.
        
        Args:
            train_data: Training DataFrame with 'features' and 'label' columns
            input_size: Number of input features
            output_size: Number of output classes
            
        Returns:
            self
        """
        self.architectures = self._create_diverse_architectures(input_size, output_size)
        
        print(f"\nTraining Conglomerate Neural Network Ensemble ({self.num_models} models)...")
        print("=" * 80)
        
        for i, layers in enumerate(self.architectures):
            print(f"\nJury Member {i+1}/{self.num_models}:")
            print(f"  Architecture: {layers}")
            
            # Create trainer with unique seed for diversity
            trainer = MultilayerPerceptronClassifier(
                maxIter=self.max_iter,
                layers=layers,
                blockSize=128,
                seed=self.seed + i  # Different seed for each model
            )
            
            # Train the model
            model = trainer.fit(train_data)
            self.models.append(model)
            print(f"  ✓ Training complete")
        
        print("\n" + "=" * 80)
        print(f"Conglomerate ensemble training complete!")
        return self
    
    def predict(self, test_data):
        """
        Make predictions using the conglomerate ensemble.
        Uses majority voting for the final prediction.
        
        Args:
            test_data: Test DataFrame with 'features' column
            
        Returns:
            DataFrame with predictions from all models and final ensemble prediction
        """
        if not self.models:
            raise ValueError("Models not trained. Call fit() first.")
        
        # Get predictions from each model
        result = test_data
        
        for i, model in enumerate(self.models):
            predictions = model.transform(test_data)
            result = result.withColumn(f"prediction_{i}", predictions["prediction"])
        
        # Create prediction columns list for voting
        pred_cols = [f"prediction_{i}" for i in range(len(self.models))]
        
        # Majority voting: Select the most common prediction
        # Create an array of all predictions and find the mode
        result = result.withColumn(
            "ensemble_prediction",
            expr(f"float(array_sort(array({','.join(pred_cols)}))[{len(self.models)//2}])")
        )
        
        return result
    
    def evaluate(self, test_data, label_col="label"):
        """
        Evaluate the conglomerate ensemble and individual models.
        
        Args:
            test_data: Test DataFrame with 'features' and 'label' columns
            label_col: Name of the label column
            
        Returns:
            Dictionary with accuracy scores
        """
        predictions = self.predict(test_data)
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol="ensemble_prediction",
            metricName="accuracy"
        )
        
        ensemble_accuracy = evaluator.evaluate(predictions)
        
        # Evaluate individual models
        individual_accuracies = []
        for i, model in enumerate(self.models):
            model_predictions = model.transform(test_data)
            evaluator_individual = MulticlassClassificationEvaluator(
                labelCol=label_col,
                predictionCol="prediction",
                metricName="accuracy"
            )
            accuracy = evaluator_individual.evaluate(model_predictions)
            individual_accuracies.append(accuracy)
        
        return {
            "ensemble_accuracy": ensemble_accuracy,
            "individual_accuracies": individual_accuracies,
            "average_individual_accuracy": np.mean(individual_accuracies)
        }


def create_spark_session():
    """Create and configure Spark session for Conglomerate Neural Network."""
    spark = SparkSession.builder \
        .appName("ConglomerateNeuralNetworkExample") \
        .config("spark.mllib.neuralnetwork.enabled", "true") \
        .config("spark.mllib.neuralnetwork.conglomerate.enabled", "true") \
        .config("spark.artemis.integration.enabled", "true") \
        .getOrCreate()
    
    print("=" * 80)
    print("Conglomerate Neural Network Ensemble - Jury-Based Prediction")
    print("=" * 80)
    print(f"Spark Version: {spark.version}")
    print(f"Neural Network Support: {spark.conf.get('spark.mllib.neuralnetwork.enabled', 'false')}")
    print(f"Conglomerate Mode: {spark.conf.get('spark.mllib.neuralnetwork.conglomerate.enabled', 'false')}")
    print("=" * 80)
    
    return spark


def create_sample_data(spark):
    """Create sample data for neural network training and testing."""
    # Expanded dataset with more variety for better testing
    data = [
        (Vectors.dense([0.0, 0.0, 0.0, 0.0]), 0.0),
        (Vectors.dense([0.1, 0.1, 0.0, 0.0]), 0.0),
        (Vectors.dense([0.0, 0.0, 0.1, 0.1]), 0.0),
        (Vectors.dense([0.0, 0.0, 1.0, 1.0]), 1.0),
        (Vectors.dense([0.1, 0.1, 0.9, 0.9]), 1.0),
        (Vectors.dense([1.0, 1.0, 0.0, 0.0]), 1.0),
        (Vectors.dense([0.9, 0.9, 0.1, 0.1]), 1.0),
        (Vectors.dense([1.0, 1.0, 1.0, 1.0]), 2.0),
        (Vectors.dense([0.9, 0.9, 0.9, 0.9]), 2.0),
        (Vectors.dense([1.0, 1.0, 0.9, 0.9]), 2.0),
        (Vectors.dense([0.5, 0.5, 0.5, 0.5]), 1.0),
        (Vectors.dense([0.4, 0.6, 0.5, 0.5]), 1.0),
    ]
    
    df = spark.createDataFrame(data, ["features", "label"])
    return df


def main():
    """Main function to demonstrate Conglomerate Neural Network Ensemble."""
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Create sample data
        print("\nPreparing data...")
        all_data = create_sample_data(spark)
        
        # Split into training and test sets
        train_data, test_data = all_data.randomSplit([0.7, 0.3], seed=1234)
        
        print(f"Training samples: {train_data.count()}")
        print(f"Test samples: {test_data.count()}")
        
        # Create and train conglomerate ensemble
        conglomerate = ConglomerateNeuralNetwork(
            num_models=3,
            max_iter=100,
            seed=1234
        )
        
        conglomerate.fit(train_data, input_size=4, output_size=3)
        
        # Evaluate the ensemble
        print("\nEvaluating Conglomerate Ensemble...")
        print("=" * 80)
        
        results = conglomerate.evaluate(test_data)
        
        print("\nIndividual Jury Member Performance:")
        for i, acc in enumerate(results["individual_accuracies"]):
            print(f"  Jury Member {i+1}: {acc:.4f}")
        
        print(f"\nAverage Individual Accuracy: {results['average_individual_accuracy']:.4f}")
        print(f"Ensemble Accuracy (Voting): {results['ensemble_accuracy']:.4f}")
        
        improvement = results['ensemble_accuracy'] - results['average_individual_accuracy']
        print(f"\nEnsemble Improvement: {improvement:+.4f}")
        
        # Show sample predictions
        print("\n" + "=" * 80)
        print("Sample Predictions (showing individual votes and ensemble decision):")
        print("=" * 80)
        predictions = conglomerate.predict(test_data)
        predictions.select(
            "features", 
            "label",
            "prediction_0",
            "prediction_1", 
            "prediction_2",
            "ensemble_prediction"
        ).show(5, truncate=False)
        
        # Summary
        print("\n" + "=" * 80)
        print("Conglomerate Neural Network Summary")
        print("=" * 80)
        print("✓ Jury-based ensemble of neural networks demonstrated")
        print(f"✓ {len(conglomerate.models)} models trained with diverse architectures")
        print("✓ Majority voting prediction aggregation implemented")
        print(f"✓ Ensemble accuracy: {results['ensemble_accuracy']:.4f}")
        print("\nThe conglomerate approach combines multiple neural network")
        print("architectures to improve robustness and prediction accuracy,")
        print("similar to how a jury reaches consensus through voting.")
        print("=" * 80)
        
    finally:
        spark.stop()
        print("\nSpark session stopped.")


if __name__ == "__main__":
    main()
