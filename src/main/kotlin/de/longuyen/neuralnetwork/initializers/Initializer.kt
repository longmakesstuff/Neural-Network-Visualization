package de.longuyen.neuralnetwork.initializers

import org.nd4j.linalg.api.ndarray.INDArray

abstract class Initializer {
    companion object {
        private const val serialVersionUID: Long = 1
    }

    abstract fun initialize(layers: IntArray): MutableMap<String, INDArray>

    override fun toString(): String {
        return "Initializer {${this.javaClass.simpleName}}"
    }
}