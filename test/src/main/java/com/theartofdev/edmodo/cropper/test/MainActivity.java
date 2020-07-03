package com.theartofdev.edmodo.cropper.test;

import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.example.test.R;
import com.theartofdev.edmodo.cropper.CropImage;
import com.theartofdev.edmodo.cropper.CropImageView;

import java.io.File;
import java.net.URI;
import java.nio.file.Path;

public class MainActivity extends AppCompatActivity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    if (! Python.isStarted()) {
      Python.start(new AndroidPlatform(this));
    }
  }

  /** Start pick image activity with chooser. */
  public void onSelectImageClick(View view) {
    CropImage.activity(null).setGuidelines(CropImageView.Guidelines.ON).start(this);
  }

  @Override
  protected void onActivityResult(int requestCode, int resultCode, Intent data) {

    // handle result of CropImageActivity
    if (requestCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE) {
      CropImage.ActivityResult result = CropImage.getActivityResult(data);
      if (resultCode == RESULT_OK) {
//        PackageManager m = getPackageManager();
//        String s = getPackageName();
//        try {
//          PackageInfo p = m.getPackageInfo(s, 0);
//          s = p.applicationInfo.dataDir;
//        } catch (PackageManager.NameNotFoundException e) {
//          Log.w("yourtag", "Error Package name not found ", e);
//        }
        Python p = Python.getInstance();
        Uri uri = result.getUri();
        String ps = p.getModule("app").callAttr("_process",uri.getPath(),Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)).toString();
        ((ImageView) findViewById(R.id.quick_start_cropped_image)).setImageURI(Uri.fromFile(new File(ps)));
        Toast.makeText(
                this, "Cropping successful, Sample: " + result.getSampleSize(), Toast.LENGTH_LONG)
            .show();
      } else if (resultCode == CropImage.CROP_IMAGE_ACTIVITY_RESULT_ERROR_CODE) {
        Toast.makeText(this, "Cropping failed: " + result.getError(), Toast.LENGTH_LONG).show();
      }
    }
  }
}
