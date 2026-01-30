# Spark Neural Network Integration with Artemis1981

This document describes how to use Apache Spark's MLlib Neural Network capabilities and provides a configuration framework for integrating with the Jury1981/Artemis1981 repository for mobile broadband neuro network processing.

## Overview

This integration provides:
- **Apache Spark MLlib**: Native distributed machine learning library with neural network support (MultilayerPerceptronClassifier)
- **Configuration Framework**: Templates and examples for organizing neural network settings
- **Artemis1981 Integration Framework**: Configuration structure for linking to the external Artemis1981 repository

**Note**: The configuration properties for Artemis integration are provided as a framework for organizing integration settings. Actual integration with the Artemis1981 repository would require additional implementation based on specific integration requirements.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Apache Spark                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │               MLlib - Machine Learning                 │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │   Neural Network (ANN) Layer                     │ │  │
│  │  │   - MultilayerPerceptronClassifier               │ │  │
│  │  │   - Feed-Forward Neural Networks                 │ │  │
│  │  │   - Backpropagation Training                     │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↕
         Configuration Framework (for integration)
                            ↕
┌─────────────────────────────────────────────────────────────┐
│              Jury1981/Artemis1981 Repository                 │
│    (Integration requires additional implementation)          │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Files

### 1. Neural Network Configuration (`conf/neural-network.conf.template`)

This template file contains Spark-specific settings for neural network operations and SHA tracking:

```properties
# Enable Neural Network support
spark.mllib.neuralnetwork.enabled=true

# Default network architecture
spark.mllib.neuralnetwork.defaultLayers=3
spark.mllib.neuralnetwork.defaultActivation=sigmoid

# Training parameters
spark.mllib.neuralnetwork.learningRate=0.01
spark.mllib.neuralnetwork.maxIterations=100

# Mobile Broadband settings
spark.broadband.neuro.enabled=true
spark.broadband.neuro.batchSize=32
spark.broadband.neuro.optimizer=adam

# SHA tracking for version control
spark.neuralnet.config.version=1.0.0
spark.neuralnet.config.sha=initial
spark.neuralnet.repository.sha=HEAD
```

Copy this template to create your configuration:
```bash
cp conf/neural-network.conf.template conf/neural-network.conf
```

### 2. Artemis Integration Properties (`conf/artemis-integration.properties.template`)

This template file provides a configuration framework for organizing Artemis1981 repository connection settings with SSH and HTTPS URLs, plus SHA tracking:

**Note**: These properties serve as a configuration template. Actual integration functionality would require additional implementation.

```properties
# Repository connection (SSH and HTTPS)
artemis.repository.ssh=git@github.com:Jury1981/Artemis1981.git
artemis.repository.url=https://github.com/Jury1981/Artemis1981
artemis.integration.mode=active

# SHA tracking and version control
artemis.sync.commit.sha=HEAD
artemis.integration.version=1.0.0
artemis.integration.version.sha=initial
artemis.sync.auto.update.sha=true

# Data synchronization
artemis.data.sync.enabled=true
artemis.data.sync.frequency=300
artemis.data.sync.protocol=ssh

# Neural network coordination
artemis.neuralnet.coordination=true
artemis.neuralnet.dataflow=bidirectional

# SSH authentication
artemis.ssh.key.path=~/.ssh/id_rsa
artemis.ssh.known.hosts=~/.ssh/known_hosts
artemis.ssh.timeout=30
```

Copy this template to create your configuration:
```bash
cp conf/artemis-integration.properties.template conf/artemis-integration.properties
```

## Usage

### Basic Example

```python
from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier

# Create Spark session with Neural Network configuration
spark = SparkSession.builder \
    .appName("NeuralNetworkExample") \
    .config("spark.mllib.neuralnetwork.enabled", "true") \
    .config("spark.artemis.integration.enabled", "true") \
    .getOrCreate()

# Define neural network architecture
layers = [4, 5, 4, 3]  # input, hidden layers, output

# Create and train the model
trainer = MultilayerPerceptronClassifier(
    maxIter=100,
    layers=layers,
    blockSize=128
)

model = trainer.fit(training_data)
predictions = model.transform(test_data)
```

### Running the Integration Example

First, set up the configuration files:
```bash
# Copy configuration templates
cp conf/neural-network.conf.template conf/neural-network.conf
cp conf/artemis-integration.properties.template conf/artemis-integration.properties

# Edit configurations as needed:
# - Update artemis.ssh.key.path to point to your SSH private key
# - Replace HEAD with actual commit SHAs if you want to pin to specific versions
# - Customize repository URLs if using a fork
```

Then run the example:
```bash
# Using spark-submit with config files
./bin/spark-submit \
  --conf spark.mllib.neuralnetwork.enabled=true \
  --conf spark.artemis.integration.enabled=true \
  examples/src/main/python/mllib/neural_network_artemis_integration.py

# Or using configuration files
./bin/spark-submit \
  --properties-file conf/neural-network.conf \
  examples/src/main/python/mllib/neural_network_artemis_integration.py
```

## Neural Network Components

### Available in Spark MLlib

1. **MultilayerPerceptronClassifier**: Feedforward neural network classifier
   - Supports multiple hidden layers
   - Uses backpropagation for training
   - Optimized for distributed computing

2. **Layer Architecture**: Located in `mllib/src/main/scala/org/apache/spark/ml/ann/`
   - `Layer.scala`: Core layer implementation
   - `LossFunction.scala`: Loss functions for training
   - `BreezeUtil.scala`: Numerical operations

## Integration Features

### Mobile Broadband Neuro Network

This integration specifically supports:
- **Distributed Training**: Leverage Spark's distributed computing for large-scale neural networks
- **Real-time Processing**: Stream processing capabilities for continuous data
- **Scalability**: Handle large datasets across cluster nodes
- **Artemis1981 Coordination**: Bidirectional data flow with external repository

### Configuration Parameters

**Note**: The following custom configuration parameters are provided as a framework for organizing neural network and integration settings. Spark natively supports MultilayerPerceptronClassifier without additional configuration. The custom properties below can be used by applications that implement integration with Artemis1981.

| Parameter | Description | Purpose |
|-----------|-------------|---------|
| `spark.mllib.neuralnetwork.enabled` | Flag for neural network mode | Application-level toggle |
| `spark.mllib.neuralnetwork.defaultLayers` | Default number of layers | Application configuration |
| `spark.mllib.neuralnetwork.learningRate` | Learning rate for training | Application configuration |
| `spark.broadband.neuro.batchSize` | Batch size for training | Application configuration |
| `spark.artemis.integration.enabled` | Enable Artemis integration framework | Application-level toggle |
| `artemis.data.sync.frequency` | Sync interval in seconds | Integration framework setting |

## Performance Tuning

For optimal neural network performance:

```properties
# Increase memory for neural network operations
spark.neuralnet.executor.memory=4g
spark.neuralnet.driver.memory=2g

# Use Kryo serialization for better performance
spark.neuralnet.serializer=org.apache.spark.serializer.KryoSerializer

# Adjust batch size based on data size
spark.broadband.neuro.batchSize=64
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Increase executor and driver memory
2. **Slow Training**: Adjust batch size and learning rate
3. **Connection Issues**: Verify Artemis repository URL and network access

### Verification

To verify the integration is working:

```python
# Check configuration
print(spark.conf.get('spark.mllib.neuralnetwork.enabled'))
print(spark.conf.get('spark.artemis.integration.enabled'))
```

## References

- [Apache Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- [Spark Neural Network (ANN) Layer](https://github.com/apache/spark/tree/master/mllib/src/main/scala/org/apache/spark/ml/ann)
- [Jury1981/Artemis1981 Repository](https://github.com/Jury1981/Artemis1981)

## License

Licensed under the Apache License, Version 2.0. See LICENSE file for details.
