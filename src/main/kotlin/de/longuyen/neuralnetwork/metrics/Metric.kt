package de.longuyen.neuralnetwork.metrics

import org.nd4j.linalg.api.ndarray.INDArray

abstract class Metric {
    abstract fun compute(yTrue: INDArray, yPrediction: INDArray): Double

    override fun toString(): String {
        return this.javaClass.simpleName
    }
}