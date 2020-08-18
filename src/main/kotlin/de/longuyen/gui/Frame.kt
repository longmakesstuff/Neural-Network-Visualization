package de.longuyen.gui

import de.longuyen.ONE_COLOR
import de.longuyen.SIZE
import de.longuyen.ZERO_COLOR
import de.longuyen.neuralnetwork.NeuralNetwork
import de.longuyen.neuralnetwork.activations.Relu
import de.longuyen.neuralnetwork.activations.Sigmoid
import de.longuyen.neuralnetwork.initializers.HeInitializer
import de.longuyen.neuralnetwork.initializers.XavierInitializer
import de.longuyen.neuralnetwork.losses.CrossEntropy
import de.longuyen.neuralnetwork.metrics.BinaryAccuracy
import de.longuyen.neuralnetwork.optimizers.MomentumGradientDescent
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.awt.Color
import java.awt.Graphics
import java.awt.GridLayout
import java.awt.image.BufferedImage
import javax.swing.JFrame
import javax.swing.JPanel

class Frame(private val xs: Array<IntArray>, private val ys: Array<IntArray>) : JFrame() {
    private val leftImage = BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_ARGB)
    private val rightImage = BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_ARGB)
    private val bottomImage = BufferedImage(SIZE * 2, SIZE, BufferedImage.TYPE_INT_ARGB)
    private val rightCanvas = rightImage.graphics
    private val leftPanel = object : JPanel() {
        override fun paintComponent(g: Graphics) {
            super.paintComponent(g)
            g.drawImage(leftImage, 0, 0, null)
        }
    }
    private val rightPanel = object : JPanel() {
        override fun paintComponent(g: Graphics) {
            super.paintComponent(g)
            g.drawImage(rightImage, 0, 0, null)
        }
    }
    private val bottomPanel = object : JPanel() {
        override fun paintComponent(g: Graphics) {
            super.paintComponent(g)
            g.drawImage(bottomImage, 0, 0, null)
        }
    }
    private val layers = intArrayOf(2, 10, 10, 5, 1)
    private val neuralNetwork = NeuralNetwork(
        layers,
        XavierInitializer(),
        Relu(),
        Sigmoid(),
        CrossEntropy(),
        MomentumGradientDescent(0.1),
        BinaryAccuracy()
    )
    private val xTest: INDArray
    private val xTrain = Nd4j.createFromArray(xs).castTo(DataType.DOUBLE).div(SIZE.toDouble()).transpose()
    private val yTrain = Nd4j.createFromArray(ys).castTo(DataType.DOUBLE).transpose()

    init {
        val xTestArray = mutableListOf<DoubleArray>()
        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {
                xTestArray.add(doubleArrayOf(x.toDouble(), y.toDouble()))
            }
        }
        xTest = Nd4j.createFromArray(xTestArray.toTypedArray()).div(SIZE.toDouble()).transpose()

        setSize(SIZE * 2, SIZE * 2)
        layout = GridLayout(2, 1)
        val compositedImage = JPanel()
        compositedImage.layout = GridLayout(1, 2)
        compositedImage.add(leftPanel)
        compositedImage.add(rightPanel)
        add(compositedImage)
        add(bottomPanel)

        defaultCloseOperation = EXIT_ON_CLOSE
        isVisible = true
    }

    fun run() {
        while (true) {
            neuralNetwork.train(xTrain, yTrain, xTrain, yTrain, 1)

            visualizePrediction()
            visualizeTrainingData()
        }
    }

    private fun visualizeTrainingData() {
        for (i in xs.indices) {
            if (ys[i][0] == 1) {
                rightCanvas.color = ONE_COLOR
            } else {
                rightCanvas.color = ZERO_COLOR
            }
            val x = xs[i][0]
            val y = xs[i][1]
            rightCanvas.fillOval(x - 5, y - 5, 10, 10)
            rightCanvas.color = Color.BLACK
            rightCanvas.drawOval(x - 5, y - 5, 10, 10);
        }
        rightPanel.repaint()
    }

    private fun visualizePrediction() {
        val prediction = neuralNetwork.inference(xTest).reshape(intArrayOf(SIZE, SIZE)).toDoubleMatrix()
        for (y in prediction.indices) {
            for (x in prediction[y].indices) {
                if (prediction[y][x] > 0.5) {
                    val color = Color(100, 178, 255, 255 - ((1.0 - prediction[y][x]) * 255).toInt())
                    rightImage.setRGB(x, y, color.rgb)
                } else {
                    val color = Color(255, 0, 255, 255 - (prediction[y][x] * 255).toInt())
                    rightImage.setRGB(x, y, color.rgb)
                }
            }
        }
        rightPanel.repaint()
    }
}