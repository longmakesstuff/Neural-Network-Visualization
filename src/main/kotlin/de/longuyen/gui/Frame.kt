package de.longuyen.gui

import de.longuyen.SIZE
import de.longuyen.neuralnetwork.NeuralNetwork
import de.longuyen.neuralnetwork.activations.Relu
import de.longuyen.neuralnetwork.activations.Sigmoid
import de.longuyen.neuralnetwork.initializers.HeInitializer
import de.longuyen.neuralnetwork.losses.CrossEntropy
import de.longuyen.neuralnetwork.metrics.Accuracy
import de.longuyen.neuralnetwork.optimizers.Adam
import java.awt.Graphics
import java.awt.GridLayout
import java.awt.image.BufferedImage
import javax.swing.JFrame
import javax.swing.JPanel

class Frame : JFrame() {
    private val leftImage = BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_ARGB)
    private val rightImage = BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_ARGB)
    private val bottomImage = BufferedImage(SIZE * 2, SIZE, BufferedImage.TYPE_INT_ARGB)
    private val leftCanvas = leftImage.graphics
    private val rightCanvas = rightImage.graphics
    private val bottomCanvas = bottomImage.graphics
    private val leftPanel = object: JPanel(){
        override fun paintComponent(g: Graphics) {
            super.paintComponent(g)
            g.drawImage(leftImage, 0,0, null)
        }
    }
    private val rightPanel = object: JPanel(){
        override fun paintComponent(g: Graphics) {
            super.paintComponent(g)
            g.drawImage(rightImage, 0,0, null)
        }
    }
    private val bottomPanel = object: JPanel(){
        override fun paintComponent(g: Graphics) {
            super.paintComponent(g)
            g.drawImage(bottomImage, 0,0, null)
        }
    }
    private val layers = intArrayOf(2, 5, 5, 3, 2)
    private val neuralNetwork = NeuralNetwork(
        layers,
        HeInitializer(),
        Relu(),
        Sigmoid(),
        CrossEntropy(),
        Adam(0.1),
        Accuracy()
    )

    init {
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
}