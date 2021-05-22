import io.kjaer.compiletime.*

import org.bytedeco.javacpp.indexer.{UByteIndexer, FloatRawIndexer}
import org.bytedeco.opencv.global.opencv_core.*
import org.bytedeco.opencv.opencv_core.{Mat, Scalar, RectVector, UMat}
import org.bytedeco.opencv.global.opencv_imgproc.*
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier

import org.emergentorder.compiletime.*
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.backends.*

import java.nio.file.{Files, Paths, Path}
import io.bullet.borer.Cbor
import java.io.{ByteArrayOutputStream, File}

val tensorShapeDenotation = "Batch" ##: "Height" ##: "Width" ##: "Channel" ##: TSNil
val tensorDenotation: String & Singleton = "Image"
val ImageWidth = 224
val ImageHeight = 224
val Channels = 3
val OutputSize: Dimension = 512

val faceCascade = CascadeClassifier("cv2_cascades/haarcascades/haarcascade_frontalface_alt2.xml")

def shape(batch: Int) =
  batch #: ImageHeight #: ImageWidth #: Channels #: SNil

def predict(images: Array[Float], model: ORTModelBackend, batch: Dimension = 1, outputSize: Dimension = OutputSize) =
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

def getModel(path: Path = Paths.get("data", "model.onnx")) =
  val bytes = Files.readAllBytes(path)
  ORTModelBackend(bytes)

def distance(a: Array[Float], b: Array[Float]): Float =
  math.sqrt(a.zip(b).map((a, b) => math.pow(a - b, 2)).sum).toFloat

type Features = Map[String, Array[Float]]
val FeatureFilePath = "data/precomputed_features.cbor"

def saveFeatures(features: Features) =
  val file = File(FeatureFilePath)
  Cbor.encode(features).to(file).result

def loadFeatures: Features =
  val featureBytes = Files.readAllBytes(Paths.get(FeatureFilePath))
  Cbor.decode(featureBytes).to[Features].value

def toGrey(img: Mat) =
  val grey = UMat()
  img.copyTo(grey)
  cvtColor(grey, grey, COLOR_BGR2GRAY)
  equalizeHist(grey, grey)
  grey

def detectFaces(img: Mat): RectVector =
  val grey = toGrey(img)
  val faces = RectVector()
  faceCascade.detectMultiScale(grey, faces)
  faces  