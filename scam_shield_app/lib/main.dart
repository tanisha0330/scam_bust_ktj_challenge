import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:permission_handler/permission_handler.dart';
import 'package:vibration/vibration.dart';
import 'package:url_launcher/url_launcher.dart'; 
import 'package:shared_preferences/shared_preferences.dart'; 

void main() {
  runApp(const ScamShieldApp());
}

class ScamShieldApp extends StatelessWidget {
  const ScamShieldApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Scam Shield',
      theme: ThemeData(
        primarySwatch: Colors.teal,
        useMaterial3: true,
        scaffoldBackgroundColor: Colors.grey[50],
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.teal,
          foregroundColor: Colors.white,
          centerTitle: true,
          elevation: 2,
        ),
      ),
      home: const MainScreen(),
    );
  }
}

// --- TRANSLATION DATA ---
// Scope note: English and Hindi only. Bengali and Tamil were removed because
// the scam-detection model is trained exclusively on English and romanised
// Hinglish -- it has never seen a Bengali or Tamil scam. Offering those
// languages in the UI implied a protection level the model cannot deliver.
class _AppTranslations {
  static final Map<String, Map<String, String>> _values = {
    'app_title': {'en': 'Scam Shield', 'hi': 'स्कैम सुरक्षा'},
    'sms_tab': {'en': 'SMS Shield', 'hi': 'संदेश सुरक्षा'},
    'call_tab': {'en': 'Call Shield', 'hi': 'कॉल सुरक्षा'},
    'settings': {'en': 'Settings', 'hi': 'सेटिंग्स'},
    // SMS Screen
    'trusted_contact': {'en': 'Trusted Contact (Phone)', 'hi': 'भरोसेमंद साथी (फोन)'},
    'paste_msg': {'en': 'Paste suspicious text here...', 'hi': 'शक वाला मैसेज यहाँ पेस्ट करें...'},
    'check_btn': {'en': 'CHECK SAFETY', 'hi': 'सुरक्षा जांचें'},
    'scanning': {'en': 'Scanning...', 'hi': 'जांच जारी...'},
    'family_notified': {'en': 'Family Alerted Automatically', 'hi': 'परिवार को सूचित कर दिया गया है'},
    'safe_msg': {'en': 'SAFE MESSAGE', 'hi': 'सुरक्षित मैसेज'},
    'scam_alert': {'en': 'SCAM ALERT!', 'hi': 'स्कैम अलर्ट!'},
    'suspicious_msg': {'en': 'BE CAREFUL', 'hi': 'सावधान रहें'},
    'suspicious_advice': {
      'en': 'This could be genuine, but we are not sure. Do not pay or share any code. Check with family first.',
      'hi': 'यह असली हो सकता है, पर हमें पक्का नहीं पता। पैसे या कोई कोड न दें। पहले परिवार से पूछें।'
    },
    // Call Screen
    'call_sim_title': {'en': 'Call Simulator', 'hi': 'कॉल सिमुलेटर'},
    'enter_num': {'en': 'Enter Phone Number', 'hi': 'फोन नंबर दर्ज करें'},
    'sim_btn': {'en': 'SIMULATE INCOMING CALL', 'hi': 'कॉल शुरू करें (डेमो)'},
    'live_monitor_btn': {'en': 'START LIVE MONITOR', 'hi': 'लाइव मॉनिटर शुरू करें'},
    'protect_dialog': {'en': 'Activate AI protection?', 'hi': 'AI सुरक्षा चालू करें?'},
    'yes_protect': {'en': 'YES, PROTECT', 'hi': 'हाँ, रक्षा करें'},
    // Live Screen
    'live_shield': {'en': 'Live Shield', 'hi': 'लाइव रक्षा'},
    'listening': {'en': 'Listening...', 'hi': 'सुन रहा हूँ...'},
    'risk_safe': {'en': 'Safe', 'hi': 'सुरक्षित'},
    'risk_danger': {'en': 'DANGER', 'hi': 'खतरा'},
    'mic_denied': {'en': 'Mic Permission Denied', 'hi': 'माइक अनुमति अस्वीकृत'},
  };

  static String t(String key, String lang) {
    return _values[key]?[lang] ?? _values[key]?['en'] ?? key;
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});
  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _currentIndex = 0;
  String _currentLang = 'en'; // Default Language
  final TextEditingController ipController = TextEditingController(text: "10.145.73.107:8000");

  @override
  Widget build(BuildContext context) {
    final List<Widget> screens = [
      MessageShieldScreen(ipController: ipController, lang: _currentLang),
      CallShieldScreen(ipController: ipController, lang: _currentLang),
    ];

    return Scaffold(
      appBar: AppBar(
        title: Text(_currentIndex == 0
          ? _AppTranslations.t('sms_tab', _currentLang)
          : _AppTranslations.t('call_tab', _currentLang)
        ),
        actions: [
          // LANGUAGE
          DropdownButton<String>(
            value: _currentLang,
            dropdownColor: Colors.teal.shade700,
            icon: const Icon(Icons.language, color: Colors.white),
            underline: Container(),
            style: const TextStyle(color: Colors.white, fontSize: 14),
            items: const [
              DropdownMenuItem(value: 'en', child: Text("ENG")),
              DropdownMenuItem(value: 'hi', child: Text("हिन्दी")),
            ],
            onChanged: (val) {
              if (val != null) setState(() => _currentLang = val);
            },
          ),
          const SizedBox(width: 10),
          // SETTINGS
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () => showDialog(context: context, builder: (ctx) => AlertDialog(
              title: Text(_AppTranslations.t('settings', _currentLang)),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  TextField(controller: ipController),
                  const SizedBox(height: 10),
                  const Text("Tip: Use 'ipconfig' on PC to find IP.", style: TextStyle(fontSize: 12, color: Colors.grey)),
                ],
              ),
              actions: [TextButton(onPressed: ()=>Navigator.pop(ctx), child: const Text("OK"))],
            )),
          ),
          const SizedBox(width: 10),
        ],
      ),
      body: screens[_currentIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (index) => setState(() => _currentIndex = index),
        destinations: [
          NavigationDestination(
            icon: const Icon(Icons.message), 
            label: _AppTranslations.t('sms_tab', _currentLang)
          ),
          NavigationDestination(
            icon: const Icon(Icons.call), 
            label: _AppTranslations.t('call_tab', _currentLang)
          ),
        ],
      ),
    );
  }
}

// =========================================================
// SCREEN 1: MESSAGE SHIELD (Automatic Alert Logic)
// =========================================================
class MessageShieldScreen extends StatefulWidget {
  final TextEditingController ipController;
  final String lang;
  const MessageShieldScreen({super.key, required this.ipController, required this.lang});

  @override
  State<MessageShieldScreen> createState() => _MessageShieldScreenState();
}

class _MessageShieldScreenState extends State<MessageShieldScreen> {
  final TextEditingController _msgController = TextEditingController();
  final TextEditingController _contactController = TextEditingController();
  String _result = "";
  String _reason = "";
  Color _resultColor = Colors.grey;
  bool _isLoading = false;
  bool _isScamDetected = false;

  @override
  void initState() {
    super.initState();
    _loadContact();
  }

  Future<void> _loadContact() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _contactController.text = prefs.getString('family_contact') ?? "";
    });
  }

  Future<void> _saveContact(String value) async {
    final prefs = await SharedPreferences.getInstance();
    prefs.setString('family_contact', value);
  }

  // --- AUTOMATED ALERT SYSTEM ---
  Future<void> _triggerAutoAlert(String scamReason) async {
    String phone = _contactController.text.trim();
    if (phone.isEmpty) return; // Contact nahi hai to kuch mat karo

    String suspiciousMsg = _msgController.text;
    if (suspiciousMsg.length > 50) suspiciousMsg = suspiciousMsg.substring(0, 50) + "...";

    // Detailed Message for Family
    String alertText = 
      "🚨 *SCAM ALERT!* \n\n"
      "Namaste! Aapka number aapke parent/buzurg ne Scam Shield App mein emergency contact ke roop mein add kiya tha.\n\n"
      "Unhe abhi ek message aaya hai jo *Most Likely Scam* hai:\n"
      "-------------------\n"
      "\"$suspiciousMsg\"\n"
      "-------------------\n"
      "⚠️ *Reason:* $scamReason\n\n"
      "Kripya unhe turant call karke satark karein!";

    // Automatically launch SMS intent
    final url = Uri.parse("sms:$phone?body=${Uri.encodeComponent(alertText)}");
    if (await canLaunchUrl(url)) {
      await launchUrl(url);
    }
  }

  Future<void> checkScam() async {
    final String message = _msgController.text;
    String ipAddress = widget.ipController.text.trim();
    if (message.isEmpty) return;
    if (!ipAddress.contains(':')) ipAddress = "$ipAddress:8000";

    FocusScope.of(context).unfocus();
    if (_contactController.text.isNotEmpty) _saveContact(_contactController.text);

    setState(() { 
      _isLoading = true; 
      _result = ""; 
      _reason = "";
      _isScamDetected = false; 
      _resultColor = Colors.blue; 
    });

    try {
      final response = await http.post(
        Uri.parse('http://$ipAddress/predict'),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "message": message,
          "trusted_contact": _contactController.text.isNotEmpty ? _contactController.text : "Not Provided"
        }),
      ).timeout(const Duration(seconds: 20)); // Increased Timeout to 20s

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['is_scam'] is! bool) {
          setState(() {
            _result = "⚠️ Unexpected server response. Please try again.";
            _resultColor = Colors.orange;
          });
          return;
        }
        // Three-way verdict; fall back to the old boolean for older servers.
        final String verdict = (data['verdict'] as String?) ??
            ((data['is_scam'] as bool) ? 'scam' : 'safe');
        final bool isScam = verdict == 'scam';
        final bool isSuspicious = verdict == 'suspicious';

        String title;
        Color colour;
        if (isScam) {
          title = _AppTranslations.t('scam_alert', widget.lang);
          colour = Colors.red.shade700;
        } else if (isSuspicious) {
          title = _AppTranslations.t('suspicious_msg', widget.lang);
          colour = Colors.amber.shade800;
        } else {
          title = _AppTranslations.t('safe_msg', widget.lang);
          colour = Colors.green.shade700;
        }

        setState(() {
          _reason = (data['reason'] as String?) ?? "";
          final advice = isSuspicious
              ? "\n\n${_AppTranslations.t('suspicious_advice', widget.lang)}"
              : "";
          _result = "$title\n\n$_reason$advice";
          _resultColor = colour;
          _isScamDetected = isScam;
        });

        // Auto-alert the family contact ONLY on a confirmed scam. A
        // "suspicious" verdict means the two analysis tiers disagreed, so
        // messaging relatives about it would cry wolf on genuinely ambiguous
        // (often legitimate) financial messages.
        if (isScam && _contactController.text.isNotEmpty) {
          _triggerAutoAlert(_reason);
        }

      } else {
        setState(() { _result = "Server Error: ${response.statusCode}"; _resultColor = Colors.orange; });
      }
    } on TimeoutException catch (_) {
      setState(() { 
        _result = "⏱️ Server Timeout!\n\nAI took too long to respond. Check if the server is running."; 
        _resultColor = Colors.orange; 
      });
    } catch (e) {
      String errorMsg = "Connection Failed.";
      if (e.toString().contains("SocketException") || e.toString().contains("No route to host")) {
        errorMsg = "❌ No Route to Host!\n\n1. Check PC IP (ipconfig).\n2. Ensure Phone & PC are on SAME WiFi.";
      } else {
        errorMsg += "\nError: $e";
      }
      setState(() { _result = errorMsg; _resultColor = Colors.redAccent; });
    } finally {
      setState(() { _isLoading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          TextField(
            controller: _contactController,
            keyboardType: TextInputType.phone,
            onChanged: _saveContact,
            decoration: InputDecoration(
              prefixIcon: const Icon(Icons.people, color: Colors.teal),
              hintText: _AppTranslations.t('trusted_contact', widget.lang),
              filled: true, fillColor: Colors.white,
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none),
            ),
          ),
          const SizedBox(height: 15),
          TextField(
            controller: _msgController,
            maxLines: 4,
            decoration: InputDecoration(
              hintText: _AppTranslations.t('paste_msg', widget.lang),
              filled: true, fillColor: Colors.white,
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none),
            ),
          ),
          const SizedBox(height: 20),
          ElevatedButton.icon(
            onPressed: _isLoading ? null : checkScam,
            icon: _isLoading ? const SizedBox() : const Icon(Icons.shield),
            label: Text(
              _isLoading ? _AppTranslations.t('scanning', widget.lang) : _AppTranslations.t('check_btn', widget.lang),
              style: const TextStyle(fontWeight: FontWeight.bold)
            ),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.teal, foregroundColor: Colors.white,
              padding: const EdgeInsets.all(15),
              textStyle: const TextStyle(fontSize: 18)
            ),
          ),
          const SizedBox(height: 20),
          if (_result.isNotEmpty) Container(
            padding: const EdgeInsets.all(15),
            decoration: BoxDecoration(
              color: Colors.white, border: Border.all(color: _resultColor, width: 2),
              borderRadius: BorderRadius.circular(10)
            ),
            child: Column(children: [
              Icon(
                _resultColor == Colors.red.shade700
                    ? Icons.warning
                    : _resultColor == Colors.amber.shade800
                        ? Icons.help_outline          // suspicious: caution, not "all clear"
                        : (_resultColor == Colors.redAccent || _resultColor == Colors.orange)
                            ? Icons.wifi_off
                            : Icons.check_circle,
                size: 50,
                color: _resultColor,
              ),
              const SizedBox(height: 10),
              Text(_result, textAlign: TextAlign.center, style: TextStyle(fontSize: 16, color: _resultColor, fontWeight: FontWeight.bold)),
              
              if (_isScamDetected && _contactController.text.isNotEmpty) ...[
                const SizedBox(height: 20),
                const Divider(),
                // User feedback that action was taken
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(color: Colors.green.shade50, borderRadius: BorderRadius.circular(5)),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.check_circle, color: Colors.green, size: 16),
                      const SizedBox(width: 5),
                      Text("Auto-Alert prepared for family", style: TextStyle(color: Colors.green.shade800, fontWeight: FontWeight.bold)),
                    ],
                  ),
                )
              ]
            ]),
          )
        ],
      ),
    );
  }
}

// =========================================================
// SCREEN 2: CALL SHIELD
// =========================================================
class CallShieldScreen extends StatefulWidget {
  final TextEditingController ipController;
  final String lang;
  const CallShieldScreen({super.key, required this.ipController, required this.lang});

  @override
  State<CallShieldScreen> createState() => _CallShieldScreenState();
}

class _CallShieldScreenState extends State<CallShieldScreen> {
  final TextEditingController _phoneController = TextEditingController();

  Future<void> simulateIncomingCall() async {
    String phone = _phoneController.text.trim();
    if (phone.isEmpty) return;
    
    // NOTE: For demo simplicity, we allow bypassing server check if offline
    // but ideally we check server.
    String ipAddress = widget.ipController.text.trim();
    if (!ipAddress.contains(':')) ipAddress = "$ipAddress:8000";

    try {
      final response = await http.post(
        Uri.parse('http://$ipAddress/check_number'),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"phone": phone}),
      ).timeout(const Duration(seconds: 10)); // Increased Timeout to 10s

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final bool isSpam = data['is_known_spam'] == true;
        final String message = (data['message'] as String?) ?? "Unknown Caller (Server Verified)";
        _showProtectDialog(phone, message, isSpam);
      }
    } on TimeoutException catch (_) {
      // Fallback if timeout
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Timeout: Assuming Unknown Caller")));
      _showProtectDialog(phone, "Unknown Caller (Offline)", false);
    } catch (e) {
      // Fallback if offline
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Offline Mode: Assuming Unknown Caller")));
      _showProtectDialog(phone, "Unknown Caller (Offline)", false);
    }
  }

  void _showProtectDialog(String phone, String message, bool isSpam) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        backgroundColor: isSpam ? Colors.red.shade50 : Colors.white,
        title: Row(
          children: [
            Icon(isSpam ? Icons.warning_amber : Icons.help_outline, color: isSpam ? Colors.red : Colors.orange, size: 30),
            const SizedBox(width: 10),
            const Expanded(child: Text("Scam Shield", style: TextStyle(fontWeight: FontWeight.bold))),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text("Incoming Call: $phone", style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            Text(message),
            const SizedBox(height: 10),
            Text(_AppTranslations.t('protect_dialog', widget.lang)),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text("No")),
          ElevatedButton.icon(
            icon: const Icon(Icons.mic),
            label: Text(_AppTranslations.t('yes_protect', widget.lang)),
            style: ElevatedButton.styleFrom(backgroundColor: Colors.teal, foregroundColor: Colors.white),
            onPressed: () {
              Navigator.pop(context);
              Navigator.push(context, MaterialPageRoute(builder: (context) => LiveCallScreen(
                ipController: widget.ipController,
                lang: widget.lang,
              )));
            },
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(30),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.add_call, size: 80, color: Colors.teal),
            const SizedBox(height: 20),
            Text(_AppTranslations.t('call_sim_title', widget.lang), style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
            const SizedBox(height: 30),
            TextField(
              controller: _phoneController,
              keyboardType: TextInputType.phone,
              textAlign: TextAlign.center,
              decoration: InputDecoration(
                hintText: _AppTranslations.t('enter_num', widget.lang),
                filled: true, fillColor: Colors.white,
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(15)),
              ),
            ),
            const SizedBox(height: 30),
            SizedBox(
              width: double.infinity,
              height: 55,
              child: ElevatedButton(
                onPressed: simulateIncomingCall,
                style: ElevatedButton.styleFrom(backgroundColor: Colors.indigo, foregroundColor: Colors.white),
                child: Text(_AppTranslations.t('sim_btn', widget.lang), style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              ),
            ),
            const SizedBox(height: 20),
            OutlinedButton.icon(
              onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (context) => LiveCallScreen(
                ipController: widget.ipController,
                lang: widget.lang,
              ))),
              icon: const Icon(Icons.record_voice_over),
              label: Text(_AppTranslations.t('live_monitor_btn', widget.lang)),
              style: OutlinedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 20)),
            ),
          ],
        ),
      ),
    );
  }
}

// =========================================================
// SCREEN 3: LIVE CALL MONITOR
// =========================================================
class LiveCallScreen extends StatefulWidget {
  final TextEditingController ipController;
  final String lang;
  const LiveCallScreen({super.key, required this.ipController, required this.lang});

  @override
  State<LiveCallScreen> createState() => _LiveCallScreenState();
}

class _LiveCallScreenState extends State<LiveCallScreen> with SingleTickerProviderStateMixin {
  late stt.SpeechToText _speech;
  String _text = "";
  String _riskStatus = "Safe";
  double _riskScore = 0.0;
  late AnimationController _animController;
  bool _permissionDenied = false;

  @override
  void initState() {
    super.initState();
    _speech = stt.SpeechToText();
    _text = _AppTranslations.t('listening', widget.lang);
    _riskStatus = _AppTranslations.t('risk_safe', widget.lang);
    _animController = AnimationController(vsync: this, duration: const Duration(seconds: 1))..repeat(reverse: true);
    _initSpeech();
  }

  void _initSpeech() async {
    var status = await Permission.microphone.status;
    if (!status.isGranted) status = await Permission.microphone.request();

    if (status != PermissionStatus.granted) {
      if (mounted) setState(() { _text = _AppTranslations.t('mic_denied', widget.lang); _permissionDenied = true; });
      return;
    }

    bool available = await _speech.initialize(
      onStatus: (val) { if (val == 'done' && mounted && !_permissionDenied) _startListening(); },
      onError: (val) => print('Speech Error: $val'),
    );

    if (available) _startListening();
    else if (mounted) setState(() => _text = "Speech unavailable");
  }

  void _startListening() {
    _speech.listen(
      onResult: (val) {
        if (mounted) {
          setState(() {
            _text = val.recognizedWords.isEmpty ? _AppTranslations.t('listening', widget.lang) : val.recognizedWords;
            if (val.finalResult) _analyzeText(_text);
          });
        }
      },
      listenFor: const Duration(seconds: 30),
      pauseFor: const Duration(seconds: 5),
      partialResults: true,
      localeId: "en_IN",
    );
  }

  Future<void> _analyzeText(String transcript) async {
    if (transcript.isEmpty) return;
    String ip = widget.ipController.text.trim();
    if (!ip.contains(':')) ip = "$ip:8000";

    try {
      final response = await http.post(
        Uri.parse('http://$ip/analyze_call'),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "transcript": transcript,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        String action = data['action'] ?? 'none';
        if (mounted) setState(() => _riskScore = (data['risk_score'] ?? 0).toDouble());

        if (action == 'vibrate_strong') {
          if (mounted) setState(() => _riskStatus = "⚠️ ${_AppTranslations.t('risk_danger', widget.lang)}");
          if (await Vibration.hasVibrator() ?? false) Vibration.vibrate(pattern: [500, 1000, 500, 1000]); 
        } else if (action == 'vibrate_gentle') {
          if (mounted) setState(() => _riskStatus = "Suspicious...");
        } else {
           if (mounted) setState(() => _riskStatus = _AppTranslations.t('risk_safe', widget.lang));
        }
      }
    } catch (e) { print("Server Error: $e"); }
  }

  @override
  void dispose() {
    _speech.stop();
    _animController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _riskScore > 80 ? Colors.red.shade900 : Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.transparent, 
        elevation: 0,
        title: Text(_AppTranslations.t('live_shield', widget.lang), style: const TextStyle(color: Colors.white)),
        leading: BackButton(color: Colors.white, onPressed: () => Navigator.pop(context)),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            ScaleTransition(
              scale: Tween(begin: 1.0, end: 1.1).animate(_animController),
              child: Container(
                padding: const EdgeInsets.all(30),
                decoration: BoxDecoration(
                  shape: BoxShape.circle, 
                  color: _riskScore > 80 ? Colors.red : Colors.teal.withOpacity(0.3)
                ),
                child: Icon(_riskScore > 80 ? Icons.warning : Icons.mic, color: Colors.white, size: 50),
              ),
            ),
            const SizedBox(height: 30),
            Text(_riskStatus, style: const TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            LinearProgressIndicator(value: _riskScore / 100, color: _riskScore > 80 ? Colors.yellow : Colors.green, backgroundColor: Colors.grey),
            if (_permissionDenied)
              Padding(
                padding: const EdgeInsets.only(top: 20),
                child: ElevatedButton(onPressed: () => openAppSettings(), child: const Text("Open Settings to Enable Mic")),
              ),
            const SizedBox(height: 30),
            Expanded(
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(15),
                decoration: BoxDecoration(color: Colors.white10, borderRadius: BorderRadius.circular(15)),
                child: SingleChildScrollView(
                  reverse: true,
                  child: Text(_text, style: const TextStyle(color: Colors.white70, fontSize: 18)),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
