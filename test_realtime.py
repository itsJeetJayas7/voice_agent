import asyncio, json, base64, numpy as np, websockets, librosa
from vllm.assets.audio import AudioAsset

audio_path = str(AudioAsset('mary_had_lamb').get_local_path())
audio, _ = librosa.load(audio_path, sr=16000, mono=True)
pcm16 = (audio * 32767).astype(np.int16)
audio_bytes = pcm16.tobytes()
print(f'Audio: {len(audio)/16000:.1f}s')

async def test():
    uri = 'ws://localhost:8000/v1/realtime'
    async with websockets.connect(uri) as ws:
        resp = json.loads(await ws.recv())
        print(f'Session: {resp["type"]} ({resp["id"]})')
        await ws.send(json.dumps({'type': 'session.update', 'model': 'mistralai/Voxtral-Mini-4B-Realtime-2602'}))
        await ws.send(json.dumps({'type': 'input_audio_buffer.commit'}))
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i+chunk_size]
            b64 = base64.b64encode(chunk).decode('utf-8')
            await ws.send(json.dumps({'type': 'input_audio_buffer.append', 'audio': b64}))
        await ws.send(json.dumps({'type': 'input_audio_buffer.commit', 'final': True}))
        print('Audio sent. Waiting for transcription...')
        transcript = ''
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=120)
                data = json.loads(msg)
                t = data.get('type', '')
                if t == 'transcription.delta':
                    transcript += data['delta']
                    print(f'DELTA: {repr(data["delta"])}', flush=True)
                elif t == 'transcription.done':
                    print(f'DONE: {repr(data.get("text",""))}')
                    break
                elif t == 'error':
                    print(f'ERROR: {data}')
                    break
                else:
                    print(f'Event: {t}')
        except asyncio.TimeoutError:
            print(f'Timeout after 120s. Transcript so far: {repr(transcript)}')

asyncio.run(test())
