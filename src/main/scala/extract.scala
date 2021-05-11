import io.kjaer.compiletime.*
import org.emergentorder.compiletime.*
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.backends.*
import org.bytedeco.opencv.global.opencv_imgcodecs.*
import org.bytedeco.opencv.opencv_core.Mat
import io.bullet.borer.Cbor

import java.io.{ByteArrayOutputStream, File}
import java.nio.file.{Files, Paths}
import javax.imageio.ImageIO
import scala.collection.parallel.CollectionConverters.*


def label2Features(dirs: Array[File]) = 
  val model = getModel  
  val batchSize = 16
  val outputSize: Dimension = 512

  dirs.par.map { dir =>
    val label = dir.getName
    println(s"Extractting features for '$label' at $dir folder")

    val groups = dir.listFiles.grouped(batchSize)
    val features = groups.map { files =>
      val images = files.map(f => toArray(imread(f.toString))).flatten
      val currentBatch: Dimension = files.length.asInstanceOf[Dimension]      
      println(s"batch size: $currentBatch, files: ${files.mkString(",")}")
      val input = Tensor(images, tensorDenotation, tensorShapeDenotation, shape(currentBatch))           
      val out = model.fullModel[Float, "ImageClassification", "Batch" ##: "Features" ##: TSNil, currentBatch.type #: outputSize.type #: SNil](Tuple(input))
      println(s"out size: ${out.data.length}")
      out.data.grouped(outputSize).toList
    }  
    label -> features.toList.flatten
  }

@main
def extract =
  val dirs = Paths.get("dataset-family").toFile.listFiles

  val avgFeatures = label2Features(dirs).par.map {
    (label, features) => 
      val count = features.length
      println(s"label = $label, files count = $count")
      val sum = features.reduce((a, b) => a.zip(b).map(_ + _))
      label -> sum.map(_ / count)
  }.toList.toMap

  val file = File("data/precomputed_features.cbor")
  Cbor.encode(avgFeatures).to(file).result
  

