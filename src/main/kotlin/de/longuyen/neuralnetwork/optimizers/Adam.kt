package de.longuyen.neuralnetwork.optimizers

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import java.io.Serializable
import kotlin.math.pow

/**
 * Adam (short for Adaptive Moment Estimation) is an update to the RMSProp optimizer.
 */
class Adam(private val learningRate: Double, private val alpha: Double=0.1, private val eps: Double=0.00000001, private val beta1: Double = 0.9, private val beta2: Double = 0.999): Optimizer(), Serializable{
    companion object {
        private const val serialVersionUID: Long = 1
    }
    private var initialized = false
    private val M = mutableMapOf<String, INDArray>()
    private val R = mutableMapOf<String, INDArray>()
    private var epoch = 1

    override fun optimize(weights: MutableMap<String, INDArray>, gradients: MutableMap<String, INDArray>, layers: Int) {
        if(!initialized){
            initialized = true
            for(i in 2 until layers){
                M["mW$i"] = Nd4j.zerosLike(gradients["dW$i"]!!)
                R["rW$i"] = Nd4j.zerosLike(gradients["dW$i"]!!)
                M["mb$i"] = Nd4j.zerosLike(gradients["db$i"]!!)
                R["rb$i"] = Nd4j.zerosLike(gradients["db$i"]!!)
            }
        }
        for(i in 2 until layers){
            M["mW$i"] = (M["mW$i"]!!.mul(beta1)).add(gradients["dW$i"]!!.mul(1.0 - beta1))
            R["rW$i"] = (R["rW$i"]!!.mul(beta2)).add((Transforms.pow(gradients["dW$i"]!!, 2.0)).mul(1.0 - beta2))

            val wMHat = M["mW$i"]!!.div(1.0 - beta1.pow(epoch))
            val wRHat = R["rW$i"]!!.div(1.0 - beta2.pow(epoch))
            weights["W$i"] = weights["W$i"]!!.sub(((wMHat.mul(alpha)).div(Transforms.sqrt(wRHat).add(eps))).mul(learningRate))

            M["mb$i"] = (M["mb$i"]!!.mul(beta1)).add(gradients["db$i"]!!.mul(1.0 - beta1))
            R["rb$i"] = (R["rb$i"]!!.mul(beta2)).add((Transforms.pow(gradients["db$i"]!!, 2.0)).mul(1.0 - beta2))

            val bMHat = M["mb$i"]!!.div(1.0 - beta1.pow(epoch))
            val bRHat = R["rb$i"]!!.div(1.0 - beta2.pow(epoch))
            weights["b$i"] = weights["b$i"]!!.sub(((bMHat.mul(alpha)).div(Transforms.sqrt(bRHat).add(eps))).mul(learningRate))
        }
        epoch++
    }

}