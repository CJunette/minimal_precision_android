package com.example.minimal_precision_exp;

import android.animation.Animator;
import android.animation.ArgbEvaluator;
import android.animation.ValueAnimator;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.ShapeDrawable;
import android.hardware.Camera;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Objects;

public class ShowCirclesAndRecordData
{
    static final int mDurationGradient = 1000;
    private final MainActivity mContext;
    ArrayList<CircleClass> mCircles;
    ViewGroup mLayout;
    Integer mCircleIndex;
    Integer mNextCircleIndex;
    Integer mOrderIndex;
    boolean mBoolColorChanging = false;
    Integer mPictureNumber;
    String mCurrentCircleName;

    public ShowCirclesAndRecordData(MainActivity context, ViewGroup layout)
    {
        mContext = context;
        mLayout = layout;
        mCircles = prepareCircles();
        mOrderIndex = -1;
        mCircleIndex = -1;
        mNextCircleIndex = -1;
//        Collections.shuffle(mCircles);

        for (CircleClass circle : mCircles)
        {
            layout.addView(circle.mCircleView); // 将圆形添加到布局中
            circle.mCircleView.setVisibility(View.INVISIBLE); // 设置圆形不可见
        }
    }

    public ArrayList<CircleClass> prepareCircles()
    {
        ArrayList<CircleClass> circles = new ArrayList<CircleClass>();

        // resolution is 1080px(width), 2312px(height)
        // size is 6.7cm(width), 14.3cm(height)
        // 1.0cm=160px
        int circle_dist = 160;
        int circle_width = 20;
        int width = 1080;
        int height = 2312;
        int num_width = (int) (width / circle_dist);
//        int num_width = 2; // For debug
        int num_height = (int) (height / circle_dist);
//        int num_height = 0; // For debug
        int padding_width = width - num_width * circle_dist;
        int padding_height = height - num_height * circle_dist;

        for (int j = 0; j < num_height + 1; j++)
        {
            for (int i = 0; i < num_width + 1; i++)
            {
                View circleView = new View(mContext);
                ViewGroup.LayoutParams params = new ViewGroup.LayoutParams(circle_width, circle_width); // 设置圆的大小
                circleView.setLayoutParams(params);
                circleView.setBackgroundResource(R.drawable.circle); // 设置圆形背景

                // 设置圆形的位置（假设是绝对位置）
                circleView.setX((int) (padding_width / 2 - circle_width / 2) + i * circle_dist); // X坐标
                circleView.setY((int) (padding_height / 2 - circle_width / 2) + j * circle_dist); // Y坐标

                //                Log.e("location", "x: " + circleView.getX() + ", y: " + circleView.getY());

                CircleClass circle_class = new CircleClass(circleView, String.format("row_%d-col_%d", j, i));
                circles.add(circle_class);
            }
        }

        return circles;
    }

    void setViewColorGradient(View current_view)
    {
        ValueAnimator colorAnimation = ValueAnimator.ofObject(new ArgbEvaluator(), Color.BLACK, Color.RED);
        colorAnimation.setDuration(mDurationGradient); // 设置动画持续时间为1秒

        Drawable background = current_view.getBackground();

        colorAnimation.addUpdateListener(new ValueAnimator.AnimatorUpdateListener()
        {
            @Override
            public void onAnimationUpdate(ValueAnimator animator)
            {
                mBoolColorChanging = true;
                background.setTint((int) animator.getAnimatedValue());
            }
        });

        colorAnimation.addListener(new Animator.AnimatorListener()
        {
            @Override
            public void onAnimationStart(Animator animation)
            {}

            @Override
            public void onAnimationEnd(Animator animation)
            {
                current_view.setVisibility(View.INVISIBLE);
                mBoolColorChanging = false;
                if (!Objects.equals(mNextCircleIndex, null))
                {
                    mCircles.get(mNextCircleIndex).mCircleView.setVisibility(View.VISIBLE);
                }
                background.setTint(Color.BLACK);
                mContext.mDataCommunication.sendMessage("finish");
            }

            @Override
            public void onAnimationCancel(Animator animation) {}

            @Override
            public void onAnimationRepeat(Animator animation) {}
        });
        // 启动动画
        colorAnimation.start();
    }

    boolean showNextCircles()
    {
        if (mBoolColorChanging)
        {
            return false;
        }
        if (mOrderIndex == -1)
        {
//            mCircleIndex += 1;
            mCircles.get(mNextCircleIndex).mCircleView.setVisibility(View.VISIBLE);
            mContext.mDataCommunication.sendMessage("finish");
            return true;
        }
        else
        {
            mCurrentCircleName = mCircles.get(mCircleIndex).mCircleName;
            // 点颜色渐变。
            setViewColorGradient(mCircles.get(mCircleIndex).mCircleView);
            mCurrentCircleName = null;
            if (mOrderIndex < mCircles.size())
            {
//                mCircleIndex++;
                return true;
            } else
            {
                return false;
            }
        }
    }

    boolean showLastCircles()
    {
        if (mBoolColorChanging)
        {
            return false;
        }
        if (mOrderIndex < 0)
        {
            mContext.mDataCommunication.sendMessage("finish");
            return false;
        } else
        {
            // 这里我不想在设置一个用来记录上一个index的变量，所以直接遍历整个arraylist，把所有对象都变成invisible。
            for (int i = 0; i < mCircles.size(); i++)
            {
                mCircles.get(i).mCircleView.setVisibility(View.INVISIBLE);
            }
//            mCircleIndex--;

            if (mOrderIndex >= 0)
            {
                mCircles.get(mCircleIndex).mCircleView.setVisibility(View.VISIBLE);
                mContext.mDataCommunication.sendMessage("finish");
                return true;
            } else
            {
                mContext.mDataCommunication.sendMessage("finish");
                return false;
            }
        }
    }


}
