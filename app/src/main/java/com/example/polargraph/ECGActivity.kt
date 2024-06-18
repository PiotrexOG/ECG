package com.example.polargraph

import android.os.Bundle
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContentProviderCompat.requireContext
import com.androidplot.xy.BoundaryMode
import com.androidplot.xy.StepMode
import com.androidplot.xy.XYPlot
import com.polar.sdk.api.PolarBleApi
import com.polar.sdk.api.PolarBleApiCallback
import com.polar.sdk.api.PolarBleApiDefaultImpl.defaultImplementation
import com.polar.sdk.api.errors.PolarInvalidArgument
import com.polar.sdk.api.model.PolarDeviceInfo
import com.polar.sdk.api.model.PolarEcgData
import com.polar.sdk.api.model.PolarHrData
import com.polar.sdk.api.model.PolarSensorSetting
import io.reactivex.rxjava3.android.schedulers.AndroidSchedulers
import io.reactivex.rxjava3.disposables.Disposable
import java.io.BufferedReader
import java.io.DataOutputStream
import java.io.IOException
import java.net.Socket
import java.nio.ByteBuffer
import java.util.*
import java.util.concurrent.Executors

import java.io.File
import java.io.FileWriter
import java.io.BufferedWriter
import java.io.FileReader


class ECGActivity : AppCompatActivity(), PlotterListener {
    companion object {
        private const val TAG = "ECGActivity"
    }

    private lateinit var api: PolarBleApi
    private lateinit var textViewHR: TextView
    private lateinit var textViewRR: TextView
    private lateinit var textViewDeviceId: TextView
    private lateinit var textViewBattery: TextView
    private lateinit var textViewFwVersion: TextView
    private lateinit var plot: XYPlot
    private lateinit var ecgPlotter: EcgPlotter
    private var ecgDisposable: Disposable? = null
    private var hrDisposable: Disposable? = null
    private val SERVER_PORT = 12345

    private lateinit var deviceId: String
    private lateinit var serverIP: String

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ecg)
        deviceId = intent.getStringExtra("id") ?: throw Exception("ECGActivity couldn't be created, no deviceId given")
        serverIP = intent.getStringExtra("ip") ?: throw Exception("ECGActivity couldn't be created, no serverIp given")
        textViewHR = findViewById(R.id.hr)
        textViewRR = findViewById(R.id.rr)
        textViewDeviceId = findViewById(R.id.deviceId)
        textViewBattery = findViewById(R.id.battery_level)
        textViewFwVersion = findViewById(R.id.fw_version)
        plot = findViewById(R.id.plot)

        api = defaultImplementation(
            applicationContext,
            setOf(
                PolarBleApi.PolarBleSdkFeature.FEATURE_POLAR_ONLINE_STREAMING,
                PolarBleApi.PolarBleSdkFeature.FEATURE_BATTERY_INFO,
                PolarBleApi.PolarBleSdkFeature.FEATURE_DEVICE_INFO
            )
        )
        api.setApiCallback(object : PolarBleApiCallback() {
            override fun blePowerStateChanged(powered: Boolean) {
                Log.d(TAG, "BluetoothStateChanged $powered")
            }

            override fun deviceConnected(polarDeviceInfo: PolarDeviceInfo) {
                Log.d(TAG, "Device connected " + polarDeviceInfo.deviceId)
                Toast.makeText(applicationContext, R.string.connected, Toast.LENGTH_SHORT).show()
            }

            override fun deviceConnecting(polarDeviceInfo: PolarDeviceInfo) {
                Log.d(TAG, "Device connecting ${polarDeviceInfo.deviceId}")
            }

            override fun deviceDisconnected(polarDeviceInfo: PolarDeviceInfo) {
                Log.d(TAG, "Device disconnected ${polarDeviceInfo.deviceId}")
            }

            override fun bleSdkFeatureReady(identifier: String, feature: PolarBleApi.PolarBleSdkFeature) {
                Log.d(TAG, "feature ready $feature")

                when (feature) {
                    PolarBleApi.PolarBleSdkFeature.FEATURE_POLAR_ONLINE_STREAMING -> {
                        streamECG()
                        streamHR()
                    }
                    else -> {}
                }
            }

            override fun disInformationReceived(identifier: String, uuid: UUID, value: String) {
                if (uuid == UUID.fromString("00002a28-0000-1000-8000-00805f9b34fb")) {
                    val msg = "Firmware: " + value.trim { it <= ' ' }
                    Log.d(TAG, "Firmware: " + identifier + " " + value.trim { it <= ' ' })
                    textViewFwVersion.append(msg.trimIndent())
                }
            }

            override fun batteryLevelReceived(identifier: String, level: Int) {
                Log.d(TAG, "Battery level $identifier $level%")
                val batteryLevelText = "Battery level: $level%"
                textViewBattery.append(batteryLevelText)
            }

            override fun hrNotificationReceived(identifier: String, data: PolarHrData.PolarHrSample) {
                // deprecated
            }

            override fun polarFtpFeatureReady(identifier: String) {
                // deprecated
            }

            override fun streamingFeaturesReady(identifier: String, features: Set<PolarBleApi.PolarDeviceDataType>) {
                // deprecated
            }

            override fun hrFeatureReady(identifier: String) {
                // deprecated
            }

        })
        try {
            api.connectToDevice(deviceId)
        } catch (a: PolarInvalidArgument) {
            a.printStackTrace()
        }
        val deviceIdText = "ID: $deviceId"
        textViewDeviceId.text = deviceIdText

        ecgPlotter = EcgPlotter("ECG", 130)
        ecgPlotter.setListener(this)

        plot.addSeries(ecgPlotter.getSeries(), ecgPlotter.formatter)
        plot.setRangeBoundaries(-1.5, 1.5, BoundaryMode.FIXED)
        plot.setRangeStep(StepMode.INCREMENT_BY_FIT, 0.25)
        plot.setDomainStep(StepMode.INCREMENT_BY_VAL, 130.0)
        plot.setDomainBoundaries(0, 650, BoundaryMode.FIXED)
        plot.linesPerRangeLabel = 2
    }

    public override fun onDestroy() {
        super.onDestroy()
        ecgDisposable?.let {
            if (!it.isDisposed) it.dispose()
        }
        closeSocketIfNeeded()
        api.shutDown()
    }

    private var socket: Socket? = null
    private var outputStream: DataOutputStream? = null
    private fun openSocketIfNeeded() {
        if (socket == null || outputStream == null || socket!!.isClosed) {
            try {
                socket = Socket(serverIP, SERVER_PORT)
                outputStream = DataOutputStream(socket!!.getOutputStream())
            } catch (e: IOException) {
                Log.e(TAG, "Error opening socket: $e")
            }
        }
    }

    private fun closeSocket() {
        try {
            socket?.close()
        } catch (e: IOException) {
            Log.e(TAG, "Error closing socket: $e")
        } finally {
            socket = null
        }
    }

    private val executor = Executors.newSingleThreadExecutor()

//    private fun sendDataToServer(samples: List<PolarEcgData.PolarEcgDataSample>) {
//        executor.submit {
//            try {
//                openSocketIfNeeded()
//
//                val buffer = ByteBuffer.allocate(samples.size * (4 + 8))
//                for (obj in samples) {
//                    buffer.putFloat(obj.voltage.toFloat())
//                    buffer.putLong(obj.timeStamp)
//                }
//                val bytes = buffer.array()
//
//                outputStream?.write(bytes)
//                outputStream?.flush()
//            } catch (e: IOException) {
//                Log.e(TAG, "Error sending data: $e")
//            }
//        }
//    }

    private fun sendDataToServer(samples: List<PolarEcgData.PolarEcgDataSample>) {
        executor.submit {
            try {
                openSocketIfNeeded()

                val buffer = ByteBuffer.allocate(samples.size * (4 + 8))  // Allocate enough space for float and long
                val file = File(filesDir, "ecg_data1.csv")
                if (!file.exists()) {
                    file.createNewFile()
                }
                val fileWriter = FileWriter(file, true)
                val bufferedWriter = BufferedWriter(fileWriter)

                for (obj in samples) {
                    buffer.putFloat(obj.voltage.toFloat())
                    buffer.putLong(obj.timeStamp)
                    try {
                        bufferedWriter.write("${obj.timeStamp},${obj.voltage}\n")
                    } catch (e: IOException) {
                        Log.e(TAG, "Error writing to file: $e")
                    }
                }

                val bytes = buffer.array()
                outputStream?.write(bytes)
                outputStream?.flush()
                bufferedWriter.close()

            } catch (e: IOException) {
                Log.e(TAG, "Error sending data: $e")
            }
        }
    }



    // Wywołanie tej funkcji na końcu, aby upewnić się, że gniazdo zostanie zamknięte.
    private fun closeSocketIfNeeded() {
        if (socket != null && !socket!!.isClosed) {
            closeSocket()
        }
    }


    fun streamECG() {



        val isDisposed = ecgDisposable?.isDisposed ?: true
        if (isDisposed) {
            ecgDisposable = api.requestStreamSettings(deviceId, PolarBleApi.PolarDeviceDataType.ECG)
                .toFlowable()
                .flatMap { sensorSetting: PolarSensorSetting -> api.startEcgStreaming(deviceId, sensorSetting.maxSettings()) }
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(
                    { polarEcgData: PolarEcgData ->
                        Log.d(TAG, "ecg update")
                        sendDataToServer(polarEcgData.samples)
                        Log.d(TAG, polarEcgData.samples.size.toString())
                        var siema = false
                        for (data in polarEcgData.samples) {
                            if (!siema) {
                                Log.d(TAG, "pierwsza wartosc to: ${data.voltage}")
                                Log.d(TAG, "pierwszy timestamp to: ${data.timeStamp}")
                            }
                            siema = true
                            ecgPlotter.sendSingleSample((data.voltage.toFloat() / 1000.0).toFloat())
                        }
                    },
                    { error: Throwable ->
                        Log.e(TAG, "Ecg stream failed $error")
                        ecgDisposable = null
                    },
                    {
                        Log.d(TAG, "Ecg stream complete")
                    }
                )
        } else {
            // NOTE stops streaming if it is "running"
            ecgDisposable?.dispose()
            ecgDisposable = null
        }
    }


    fun streamHR() {
        val isDisposed = hrDisposable?.isDisposed ?: true
        if (isDisposed) {
            hrDisposable = api.startHrStreaming(deviceId)
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(
                    { hrData: PolarHrData ->



                        for (sample in hrData.samples) {
                            Log.d(TAG, "HR " + sample.hr)
                            if (sample.rrsMs.isNotEmpty()) {
                                val rrText = "(${sample.rrsMs.joinToString(separator = "ms, ")}ms)"
                                textViewRR.text = rrText
                            }

                            textViewHR.text = sample.hr.toString()

                        }
                    },
                    { error: Throwable ->
                        Log.e(TAG, "HR stream failed. Reason $error")
                        hrDisposable = null
                    },
                    { Log.d(TAG, "HR stream complete") }
                )
        } else {
            // NOTE stops streaming if it is "running"
            hrDisposable?.dispose()
            hrDisposable = null
        }
    }

    override fun update() {
        runOnUiThread { plot.redraw() }
    }

}