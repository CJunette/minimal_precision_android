package com.example.minimal_precision_exp;

import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Objects;

public class DataCommunication {
    MainActivity mContext;
    int mPort = 12346;
    Socket mSocket = null;


    DataCommunication(MainActivity context) {
        mContext = context;

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    ServerSocket serverSocket = new ServerSocket(mPort);
                    mSocket = serverSocket.accept();
//                    mSocket = new Socket("localhost", mPort);
                    BufferedReader input = new BufferedReader(new InputStreamReader(mSocket.getInputStream()));

                    while (true) {
                        String message = input.readLine();

                        Log.e("message", message);
//                        PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
//                        out.println("message received");

                        String[] message_split = message.split("\\*");

                        if (message_split.length > 1)
                        {
                            String order_index = message_split[1];
                            String circle_index = message_split[2];
                            String next_circle_index = message_split[4];

                            if (Objects.equals(order_index, "null")){mContext.mShowCirclesAndRecordData.mOrderIndex = null;}
                            else{mContext.mShowCirclesAndRecordData.mOrderIndex = Integer.parseInt(order_index);}

                            if (Objects.equals(circle_index, "null")){mContext.mShowCirclesAndRecordData.mCircleIndex = null;}
                            else{mContext.mShowCirclesAndRecordData.mCircleIndex = Integer.parseInt(circle_index);}

                            if (Objects.equals(next_circle_index, "null")){mContext.mShowCirclesAndRecordData.mNextCircleIndex = null;}
                            else{mContext.mShowCirclesAndRecordData.mNextCircleIndex = Integer.parseInt(next_circle_index);}
                        }

                        if (message.equals("capture_finish*"))
                        {
                            mContext.mShowCirclesAndRecordData.turnNextCircleVisible();
                        }

                        if (message.equals("deleting_files*"))
                        {
                            mContext.mShowCirclesAndRecordData.turnAllCircleInvisible();
                        }

                        Log.e("INDEX_INFORMATION", "mOrderIndex: " + mContext.mShowCirclesAndRecordData.mOrderIndex + ", mCircleIndex: " + mContext.mShowCirclesAndRecordData.mCircleIndex);

                        mContext.runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
//                                TextView textView = mContext.findViewById(R.id.text_view);
//                                textView.setText(message);
                                if (mContext.mShowCirclesAndRecordData != null)
                                {
                                    if (Objects.equals(message, "over"))
                                    {
                                        // TODO 对于over的情况的处理。
                                    }
                                    else if (Objects.equals(message_split[0], "s"))
                                    {
                                        mContext.mShowCirclesAndRecordData.showNextCircles();
                                    }
                                    else if (Objects.equals(message_split[0], "w"))
                                    {
                                        mContext.mShowCirclesAndRecordData.showLastCircles();
                                    }
                                }
                            }
                        });

                        if (Objects.equals(message, "end")) {
                            break;
                        }
                    }
                    mSocket.close();
                    serverSocket.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }

    void sendMessage(String message)
    {
        if (mSocket != null)
        {
            new Thread(new Runnable() {
                @Override
                public void run() {
                    OutputStream outputStream = null;
                    try
                    {
                        outputStream = mSocket.getOutputStream();
                        outputStream.write(message.getBytes());
                        outputStream.flush();

                        Log.e("DATA_COMMUNICATION", message);
                    }
                    catch (IOException e)
                    {
                        e.printStackTrace();
                    }
                }
            }).start();
        }
    }
}
