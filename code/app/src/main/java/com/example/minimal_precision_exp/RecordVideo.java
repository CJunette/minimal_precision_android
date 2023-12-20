package com.example.minimal_precision_exp;
import android.Manifest;
import android.content.ContentValues;
import android.content.pm.PackageManager;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.net.Uri;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;


public class RecordVideo {
    Camera mCamera = null;
    int mCameraId_int = -1;
    MainActivity mContext;

    CameraDevice mCameraDevice = null;
    String mCameraId = null;
    CameraCaptureSession mCameraCaptureSession = null;
    Handler mBackgroundHandler;

    RecordVideo(MainActivity context)
    {
        mContext = context;

//        for (int i = 0; i < Camera.getNumberOfCameras(); i++) {
//            Camera.CameraInfo info = new Camera.CameraInfo();
//            Camera.getCameraInfo(i, info);
//            if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
//                mCameraId_int = i;
//                break;
//            }
//        }




    }
//
//
//
//
//    public void realTimeShow()
//    {
//        SurfaceView surfaceView = mContext.findViewById(R.id.camera_preview);
//        SurfaceHolder surfaceHolder = surfaceView.getHolder();
//
//        surfaceHolder.addCallback(new SurfaceHolder.Callback() {
//            @Override
//            public void surfaceCreated(SurfaceHolder holder) {
//                // Surface创建时打开相机
//                openFrontFacingCamera();
//                setCameraPreview(holder);
//            }
//
//            @Override
//            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
//                // Surface改变时更新相机预览
//                if (holder.getSurface() == null) {
//                    return;
//                }
//                try {
//                    mCamera.stopPreview();
//                } catch (Exception e) {
//                    // 忽略停止预览时的错误
//                }
//                setCameraPreview(holder);
//            }
//
//            @Override
//            public void surfaceDestroyed(SurfaceHolder holder) {
//                // Surface销毁时释放相机资源
//                if (mCamera != null) {
//                    mCamera.release();
//                    mCamera = null;
//                }
//            }
//        });
//
//        Camera.PictureCallback pictureCallback = new Camera.PictureCallback() {
//            @Override
//            public void onPictureTaken(byte[] data, Camera camera) {
//                ContentValues values = new ContentValues();
//                values.put(MediaStore.Images.Media.DISPLAY_NAME, "IMG_" + System.currentTimeMillis()); // 图片的文件名
//                values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg"); // 图片的MIME类型
//                values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + File.separator + "exp_photo" + File.separator + mContext.mSubjectName); // 存储路径
//
//                // 获取图片的URI
//                Uri imageUri = mContext.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
//
//                try {
//                    // 获取输出流
//                    OutputStream outputStream = mContext.getContentResolver().openOutputStream(imageUri);
//                    outputStream.write(data);
//                    outputStream.close();
//                } catch (Exception e) {
//                    e.printStackTrace();
//                }
//            }
//        };
//
//        Button captureButton = mContext.findViewById(R.id.camera_button);
//        captureButton.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                // 拍照
//                mCamera.takePicture(null, null, pictureCallback);
//            }
//        });
//    }
//
//    private void openFrontFacingCamera() {
//        // 获取前置相机的ID
//        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
//        for (int i = 0; i < Camera.getNumberOfCameras(); i++) {
//            Camera.getCameraInfo(i, cameraInfo);
//            if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
//                mCameraId_int = i;
//                break;
//            }
//        }
//        if (mCameraId_int >= 0) {
//            mCamera = Camera.open(mCameraId_int);
//        }
//    }
//
//    private void setCameraPreview(SurfaceHolder holder) {
//        try {
//            mCamera.setPreviewDisplay(holder);
//            mCamera.startPreview();
//        } catch (IOException e) {
//            Log.e("CameraActivity", "Error setting up camera preview: " + e.getMessage());
//        }
//    }
//
//
//    void recordDataWithCamera()
//    {
//        mContext.mShowCirclesAndRecordData.mPictureNumber = 0;
//        if (mCameraId_int >= 0) {
//            mCamera = Camera.open(mCameraId_int);
//        }
//
//        // 设置拍照回调
//        Camera.PictureCallback pictureCallback = new Camera.PictureCallback() {
//            @Override
//            public void onPictureTaken(byte[] data, Camera camera) {
//                // 保存图片
//                savePhoto(data);
//
//                // 如果还没有拍够30张，继续拍照
//                if (mContext.mShowCirclesAndRecordData.mBoolColorChanging && mContext.mShowCirclesAndRecordData.mPictureNumber < 30) {
//                    camera.takePicture(null, null, this);
//                }
//            }
//        };
//
//        // 开始拍照
//        new Handler(Looper.getMainLooper()).post(new Runnable() {
//            @Override
//            public void run() {
//                mCamera.takePicture(null, null, pictureCallback);
//            }
//        });
//    }
//
//    private void savePhoto(byte[] data) {
//        // 保存图片逻辑，例如使用FileOutputStream
////        mContext.mShowCirclesAndRecordData.mPictureNumber += 1;
//
//        String filePath = String.format("exp_photo/%s/%s/", mContext.mSubjectName, "test");
//
//
//        File myDir = new File(Environment.getExternalStorageDirectory(), filePath);
//        if (!myDir.exists()) {
//            if (myDir.mkdirs()) {
//                Log.d("App", "目录创建成功");
//            } else {
//                Log.d("App", "目录创建失败");
//            }
//        }
//
//        File photo = new File(myDir, String.format("%d.jpg", mContext.mShowCirclesAndRecordData.mPictureNumber));
//        try (FileOutputStream fos = new FileOutputStream(photo)) {
//            fos.write(data);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }
}
