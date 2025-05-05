# language_texts.py

texts = {
    "English": {
        "title": "🎙️ AI Noise Reducer",
        "subheader": "Upload or record audio to remove background noise.",
        "upload_audio": "Upload Audio",
        "record_audio": "Record Audio",
        "choose_model": "Choose noise reduction model:",
        "process_status": "Processing... Please wait.",
        "denoising_complete": "{model} denoising completed successfully!",
        "feedback_title": "🗣️ Share Your Feedback",
        "feedback_placeholder": "Write your feedback:",
        "rating_prompt": "Rate this app (1 to 5 stars)",
        "rating_labels": ["😞 Poor", "😐 Fair", "🙂 Good", "😀 Very Good", "🌟 Excellent"],
        "previous_feedbacks": "📝 Previous Feedbacks"
    },
    "Spanish": {
        "title": "🎙️ Reductor de Ruido AI",
        "subheader": "Sube o graba audio para eliminar el ruido de fondo.",
        "upload_audio": "Subir Audio",
        "record_audio": "Grabar Audio",
        "choose_model": "Elige un modelo de reducción de ruido:",
        "process_status": "Procesando... Por favor espera.",
        "denoising_complete": "¡Reducción de ruido con {model} completada exitosamente!",
        "feedback_title": "🗣️ Comparte tus comentarios",
        "feedback_placeholder": "Escribe tus comentarios:",
        "rating_prompt": "Califica esta aplicación (1 a 5 estrellas)",
        "rating_labels": ["😞 Pobre", "😐 Regular", "🙂 Bueno", "😀 Muy Bueno", "🌟 Excelente"]
    },
    "French": {
        "title": "🎙️ Réducteur de Bruit AI",
        "subheader": "Téléchargez ou enregistrez de l'audio pour supprimer le bruit de fond.",
        "upload_audio": "Télécharger Audio",
        "record_audio": "Enregistrer Audio",
        "choose_model": "Choisissez un modèle de réduction de bruit :",
        "process_status": "Traitement en cours... Veuillez patienter.",
        "denoising_complete": "Réduction de bruit par {model} terminée avec succès !",
        "feedback_title": "🗣️ Donnez votre avis",
        "feedback_placeholder": "Écrivez votre avis :",
        "rating_prompt": "Notez cette application (1 à 5 étoiles)",
        "rating_labels": ["😞 Mauvais", "😐 Passable", "🙂 Bon", "😀 Très Bon", "🌟 Excellent"]
    },
    "Marathi": {
        "title": "🎙️ एआय आवाज कमी करणारे",
        "subheader": "पार्श्वभूमीचा आवाज काढण्यासाठी ऑडिओ अपलोड करा किंवा रेकॉर्ड करा.",
        "upload_audio": "ऑडिओ अपलोड करा",
        "record_audio": "ऑडिओ रेकॉर्ड करा",
        "choose_model": "आवाज कमी करणारे मॉडेल निवडा:",
        "process_status": "प्रक्रिया चालू आहे... कृपया थांबा.",
        "denoising_complete": "{model} द्वारे आवाज कमी करणे यशस्वी झाले!",
        "feedback_title": "🗣️ तुमचे अभिप्राय द्या",
        "feedback_placeholder": "तुमचे अभिप्राय लिहा:",
        "rating_prompt": "या अ‍ॅपला रेट करा (1 ते 5 स्टार)",
        "rating_labels": ["😞 खराब", "😐 ठीक", "🙂 चांगले", "😀 खूप चांगले", "🌟 उत्कृष्ट"]
    },
    "Hindi": {
        "title": "🎙️ एआई नॉइज़ रिड्यूसर",
        "subheader": "बैकग्राउंड नॉइज़ हटाने के लिए ऑडियो अपलोड या रिकॉर्ड करें।",
        "upload_audio": "ऑडियो अपलोड करें",
        "record_audio": "ऑडियो रिकॉर्ड करें",
        "choose_model": "शोर कम करने के लिए मॉडल चुनें:",
        "process_status": "प्रोसेस हो रहा है... कृपया प्रतीक्षा करें।",
        "denoising_complete": "{model} द्वारा नॉइज़ रिडक्शन सफलतापूर्वक पूरा हुआ!",
        "feedback_title": "🗣️ अपना फीडबैक दें",
        "feedback_placeholder": "अपना फीडबैक लिखें:",
        "rating_prompt": "इस ऐप को रेट करें (1 से 5 स्टार)",
        "rating_labels": ["😞 खराब", "😐 औसत", "🙂 अच्छा", "😀 बहुत अच्छा", "🌟 बेहतरीन"]
    }
}

# Wrapper class to enable fallback to English
class SafeTexts:
    def __init__(self, texts, default_lang="English"):
        self.texts = texts
        self.default_lang = default_lang

    def __getitem__(self, lang):
        return SafeLang(self.texts, lang, self.default_lang)

class SafeLang:
    def __init__(self, texts, lang, default_lang):
        self.lang = lang
        self.default_lang = default_lang
        self.texts = texts

    def __getitem__(self, key):
        return (self.texts.get(self.lang, {}).get(key) or
                self.texts[self.default_lang].get(key, f"[Missing: {key}]"))

# Wrap the dictionary
texts = SafeTexts(texts)
