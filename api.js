
export async function getPreview(voiceApiLabel) {
  const key = voiceApiLabel.replace(/[^a-zA-Z0-9]/g, '_');
  const path = FileSystem.cacheDirectory + 'preview_' + key + '.ogg';

  const info = await FileSystem.getInfoAsync(path);
  if (info.exists) return path;

  const res = await fetch(API_URL + '/preview?voice=' + encodeURIComponent(voiceApiLabel));
  if (!res.ok) throw new Error('Preview failed (' + res.status + ')');

  const data = await res.json();
  if (!data.audio_b64) throw new Error('No preview audio');

  await FileSystem.writeAsStringAsync(path, data.audio_b64, {
    encoding: FileSystem.EncodingType.Base64,
  });
  return path;
}
