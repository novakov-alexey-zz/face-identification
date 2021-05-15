import java.nio.file.{Files, Paths}
import org.emergentorder.onnx.Tensors.*
import org.emergentorder.onnx.backends.*
import org.emergentorder.compiletime.*
import io.kjaer.compiletime.*

import java.awt.event.KeyEvent
import javax.swing.{JFrame, WindowConstants}
import java.nio.ByteBuffer
import org.bytedeco.opencv.opencv_core.{Mat, Size, UMat, RectVector, Rect, Point, Scalar}
import org.bytedeco.opencv.global.opencv_core.*
import org.bytedeco.opencv.global.opencv_imgproc.*
import org.bytedeco.opencv.opencv_videoio.*
import org.bytedeco.opencv.global.opencv_imgcodecs.*
import org.bytedeco.javacv.{CanvasFrame, OpenCVFrameConverter}
import scala.collection.mutable.ArrayBuffer
import org.bytedeco.opencv.opencv_objdetect.*
import io.bullet.borer.Cbor

def createCavasFrame = 
  val frame = CanvasFrame("Detected Faces")
  frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
  frame.setCanvasSize(1280, 720)
  frame

def toGrey(img: Mat) =
  val gray = UMat()
  img.copyTo(gray)
  cvtColor(gray, gray, COLOR_BGR2GRAY)
  equalizeHist(gray, gray)
  gray

def calcLabel(face: Array[Float], features: Features, threshold: Int = 100) = 
  features.foldLeft("?", Float.MaxValue){ 
    case ((label, min), (l, f)) => 
      val d = distance(face, f)
      if d < threshold && d < min then (l, d)
      else (label, min)
  }._1  

@main
def demo() =            
  val faceCascade = CascadeClassifier("cv2_cascades/haarcascades/haarcascade_frontalface_alt2.xml")
  val capture = VideoCapture(0)
  val canvasFrame = createCavasFrame  
  val frame = Mat()
  val converter = OpenCVFrameConverter.ToMat()
  val model = getModel
  val features = loadFeatures

  try
    while capture.read(frame) do
      val grey = toGrey(frame)
      val faces = RectVector()
      faceCascade.detectMultiScale(grey, faces)
      
      val labels = for face <- faces.get yield
        rectangle(frame,
          Point(face.x, face.y),
          Point(face.x + face.width, face.y + face.height),
          Scalar(0, 255, 0, 1)
        )
        val crop = Rect(face.x, face.y, face.width, face.height)
        val cropped = Mat(frame, crop)
        resize(cropped, cropped, Size(224, 224))
        val image = toArray(scale(cropped))

        val faceFeatures = predict(image, model).data
        val label = calcLabel(faceFeatures, features)
        (label, crop)
      
      for (label, crop) <- labels do
        val x = math.max(crop.tl.x - 10, 0)
        val y = math.max(crop.tl.y - 10, 0)
        putText(frame, label, Point(x, y), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 255, 0, 1.0))

      canvasFrame.showImage(converter.convert(frame))                              
  finally
    capture.release
    canvasFrame.dispose