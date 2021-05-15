import io.kjaer.compiletime.*

import org.bytedeco.javacpp.indexer.{UByteIndexer, FloatRawIndexer}
import org.bytedeco.opencv.global.opencv_core.*
import org.bytedeco.opencv.opencv_core.{Mat, Scalar}

import org.emergentorder.compiletime.*
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.backends.*

import java.nio.file.{Files, Paths}
import io.bullet.borer.Cbor
import java.io.{ByteArrayOutputStream, File}

def shape(batch: Int) = 
  batch #: 224 #: 224 #: 3 #: SNil
  
val tensorShapeDenotation = "Batch" ##: "Height" ##: "Width" ##: "Channel" ##: TSNil
val tensorDenotation: String & Singleton = "Image"

val outputSize: Dimension = 512

def predict(images: Array[Float], model: ORTModelBackend, batch: Dimension = 1, outputSize: Dimension = outputSize) =
  val input = Tensor(images, tensorDenotation, tensorShapeDenotation, shape(batch))
  model.fullModel[Float, "ImageClassification", "Batch" ##: "Features" ##: TSNil, batch.type #: outputSize.type #: SNil](Tuple(input))

def scale(img: Mat): Mat =
  val out = Mat()
  img.assignTo(out, CV_32FC4)
  subtract(out, Scalar(93.5940f, 104.7624f, 129.1863f, 0f)).asMat

def toArray(mat: Mat): Array[Float] =
  val w = mat.cols
  val h = mat.rows
  val c = mat.channels
  val rowStride = w * c

  val result = new Array[Float](rowStride * h)
  val indexer = mat.createIndexer[FloatRawIndexer]()
  var off = 0
  var y = 0
  while (y < h)
    indexer.get(y, result, off, rowStride)
    off += rowStride
    y += 1
  
  result

def getModel = 
  val bytes = Files.readAllBytes(Paths.get("data", "model.onnx"))
  ORTModelBackend(bytes)

def distance(a: Array[Float], b: Array[Float]): Float = 
  math.sqrt(a.zip(b).map((a, b) => math.pow(a - b, 2)).sum).toFloat


type Features = Map[String, Array[Float]]

def saveFeatures(features: Features) =
  val file = File("data/precomputed_features.cbor")
  Cbor.encode(features).to(file).result  

def loadFeatures: Features =
  val featureBytes = Files.readAllBytes(Paths.get("data/precomputed_features.cbor"))
  Cbor.decode(featureBytes).to[Features].value