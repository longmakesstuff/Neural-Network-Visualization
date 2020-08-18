package de.longuyen

import de.longuyen.gui.Frame

fun main(){
    val data = DATA_GENERATOR.generate()
    val frame = Frame(data.first, data.second)
    frame.run()
}