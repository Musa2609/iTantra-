const fs = require('fs');
const path = require('path');
const ggwaveFactory = require('./static/js/ggwave.js');

let ggwaveInstance = null;
let ggwaveContext = null;

async function getGgwave() {
  if (!ggwaveInstance) {
    ggwaveContext = await ggwaveFactory();
    const params = ggwaveContext.getDefaultParameters();
    params.sampleRateInp = 48000;
    params.sampleRateOut = 48000;
    params.sampleRate = 48000;
    params.soundMarkerThreshold = 3;
    ggwaveInstance = ggwaveContext.init(params);
  }
  return { ggwave: ggwaveContext, instance: ggwaveInstance };
}

async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  if (command === 'encode') {
    // Usage: node ggwave_helper.js encode <hexPayload> <outputWavPath> [volume]
    const hexPayload = args[1];
    const outWav = args[2];
    const volume = parseInt(args[3] || '25', 10);

    const payloadBuf = Buffer.from(hexPayload, 'hex');
    const { ggwave, instance } = await getGgwave();

    // Use AUDIBLE_FAST (1) or AUDIBLE_NORMAL (0)
    const protocolId = ggwave.ProtocolId.GGWAVE_PROTOCOL_AUDIBLE_FAST;
    const waveBytes = ggwave.encode(instance, payloadBuf, protocolId, volume);

    // waveBytes is Int8Array containing 16-bit PCM little-endian samples at 48000 Hz
    const pcmBuf = Buffer.from(waveBytes.buffer, waveBytes.byteOffset, waveBytes.byteLength);
    const sampleCount = pcmBuf.length / 2;
    const durationSec = sampleCount / 48000;

    // Write standard 16-bit mono 48000Hz WAV file
    const header = Buffer.alloc(44);
    header.write('RIFF', 0);
    header.writeUInt32LE(36 + pcmBuf.length, 4);
    header.write('WAVE', 8);
    header.write('fmt ', 12);
    header.writeUInt32LE(16, 16); // Subchunk1Size (16 for PCM)
    header.writeUInt16LE(1, 20);  // AudioFormat (1 for PCM)
    header.writeUInt16LE(1, 22);  // NumChannels (1 mono)
    header.writeUInt32LE(48000, 24); // SampleRate
    header.writeUInt32LE(48000 * 2, 28); // ByteRate (SampleRate * NumChannels * BitsPerSample/8)
    header.writeUInt16LE(2, 32);  // BlockAlign (NumChannels * BitsPerSample/8)
    header.writeUInt16LE(16, 34); // BitsPerSample (16)
    header.write('data', 36);
    header.writeUInt32LE(pcmBuf.length, 40);

    const fullWav = Buffer.concat([header, pcmBuf]);
    fs.writeFileSync(outWav, fullWav);

    console.log(JSON.stringify({
      status: 'SUCCESS',
      sampleRate: 48000,
      sampleCount: sampleCount,
      durationSec: durationSec,
      bytesWritten: fullWav.length
    }));
  } else if (command === 'decode') {
    // Usage: node ggwave_helper.js decode <inputWavPath>
    const inputWav = args[1];
    const fileBuf = fs.readFileSync(inputWav);
    // Find data chunk
    let dataOffset = 44;
    for (let i = 12; i < fileBuf.length - 8; i++) {
      if (fileBuf.toString('ascii', i, i + 4) === 'data') {
        dataOffset = i + 8;
        break;
      }
    }
    const pcmData = fileBuf.subarray(dataOffset);

    const { ggwave, instance } = await getGgwave();
    const decodedInt8 = ggwave.decode(instance, pcmData);

    if (!decodedInt8 || decodedInt8.length === 0) {
      console.log(JSON.stringify({ status: 'NO_PACKET', hexPayload: null, text: null }));
    } else {
      const decodedBuf = Buffer.from(decodedInt8.buffer, decodedInt8.byteOffset, decodedInt8.byteLength);
      console.log(JSON.stringify({
        status: 'SUCCESS',
        hexPayload: decodedBuf.toString('hex'),
        length: decodedBuf.length,
        text: decodedBuf.toString('utf-8')
      }));
    }
  } else {
    console.error('Unknown command:', command);
    process.exit(1);
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
