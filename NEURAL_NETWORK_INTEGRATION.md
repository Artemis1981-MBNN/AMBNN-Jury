# Spark Mobile Broadband Neuro Network Integration

## Overview

This repository provides configuration templates and examples for using Apache Spark's built-in Neural Network capabilities (MLlib) and establishes a configuration framework to integrate with the [Jury1981/Artemis1981](https://github.com/Jury1981/Artemis1981) repository for mobile broadband neuro network processing.

## Quick Start

### Prerequisites
- Apache Spark 4.2.0 or later
- Python 3.8+ (for PySpark examples)
- Java 8+ or Java 11+

### Configuration

1. **Enable Neural Network Support**

   Copy the neural network configuration template:
   ```bash
   cp conf/neural-network.conf.template conf/neural-network.conf
   # Review both files before merging to avoid duplicate configurations
   # Then optionally append to spark-defaults.conf:
   cat conf/neural-network.conf >> conf/spark-defaults.conf
   ```

2. **Configure Artemis1981 Integration**

   Copy the Artemis integration properties template:
   ```bash
   cp conf/artemis-integration.properties.template conf/artemis-integration.properties
   ```
   
   The template includes SSH and HTTPS URLs, SHA tracking, and integration settings.

### Running Examples

Run the neural network integration example:

```bash
./bin/spark-submit \
  --conf spark.mllib.neuralnetwork.enabled=true \
  --conf spark.artemis.integration.enabled=true \
  examples/src/main/python/mllib/neural_network_artemis_integration.py
```

## Features

- ✅ **Neural Network Support**: Examples using Spark MLlib's MultilayerPerceptronClassifier
- ✅ **Artemis1981 Configuration**: Configuration templates for linking to Jury1981/Artemis1981 repository
- ✅ **Configuration Framework**: Structured settings for neural network and integration operations
- ✅ **Distributed Training**: Leverages Spark's built-in distributed computing capabilities
- ✅ **Documentation**: Comprehensive guides and examples for neural network usage

## Architecture

```
Apache Spark MLlib
    ↓
Neural Network Layer (ANN)
    ↓
Mobile Broadband Processing
    ↓
Artemis1981 Integration
```

## Configuration Files

| File | Purpose |
|------|---------|
| `conf/neural-network.conf.template` | Neural network, mobile broadband settings, and SHA tracking |
| `conf/artemis-integration.properties.template` | Artemis1981 repository integration with SSH/HTTPS URLs and SHA tracking |
| `examples/src/main/python/mllib/neural_network_artemis_integration.py` | Example implementation |

## Documentation

Detailed documentation is available in:
- [Neural Network Artemis Integration Guide](docs/neural-network-artemis-integration.md)
- [Apache Spark README](README.md)

## Integration Points

### Spark Neural Network (MLlib)
- MultilayerPerceptronClassifier for feedforward neural networks
- Distributed training with backpropagation
- Customizable layer architecture

### Artemis1981 Repository
- Repository URL: https://github.com/Jury1981/Artemis1981
- Bidirectional data synchronization
- Enhanced neuro network capabilities

## Building

Build Spark with neural network support:

```bash
./build/mvn -DskipTests clean package
```

## Testing

Test the neural network integration:

```bash
# Run Python example
./bin/spark-submit examples/src/main/python/mllib/neural_network_artemis_integration.py

# Verify configuration
./bin/spark-submit --conf spark.mllib.neuralnetwork.enabled=true \
  --conf spark.artemis.integration.enabled=true \
  --dry-run examples/src/main/python/mllib/neural_network_artemis_integration.py
```

## Configuration Parameters

Key parameters for neural network and Artemis integration:

```properties
# Neural Network
spark.mllib.neuralnetwork.enabled=true
spark.mllib.neuralnetwork.learningRate=0.01
spark.mllib.neuralnetwork.maxIterations=100

# Mobile Broadband
spark.broadband.neuro.enabled=true
spark.broadband.neuro.batchSize=32

# Artemis Integration
spark.artemis.repository=https://github.com/Jury1981/Artemis1981
spark.artemis.integration.enabled=true
artemis.data.sync.enabled=true
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) file for details.

## Links

- [Apache Spark Official Site](https://spark.apache.org/)
- [Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- [Jury1981/Artemis1981 Repository](https://github.com/Jury1981/Artemis1981)
- [Spark Mobile Broadband Neuro Network](https://github.com/Jury1981/spark--Mobile-Broadband-Neuro-Network)
