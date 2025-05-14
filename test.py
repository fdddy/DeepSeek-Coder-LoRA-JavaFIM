import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import numpy as np
import json

model_path = "/root/autodl-tmp/deepseek-coder-1.3b-base"

tokenizer = AutoTokenizer.from_pretrained(model_path)
print(type(tokenizer))
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda()

def load_model(base_model_path, lora_path=None):
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto")

    if lora_path:
        print("Merging LoRA adapter...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # 将 LoRA 合并到模型中，方便推理

    return tokenizer, model

lora_model = "/root/LLaMA-Factory/lora_deepseekcode_fim/saves/deepseek-coder-1.3b/lora/fim_full"

tokenizer, model = load_model(model_path, lora_model)

input_text = "<｜fim▁begin｜>import com.android.volley.toolbox.JsonObjectRequest;\nimport com.example.android.mutiarabahari.R;\n\nimport org.json.JSONException;\nimport org.json.JSONObject;\n\npublic class LoginActivity extends AppCompatActivity{\n    private static final String KEY_STATUS = \"status\";\n    private static final String KEY_MESSAGE = \"message\";\n    private static final String KEY_FULL_NAME = \"full_name\";\n    private static final String KEY_USERNAME = \"username\";\n    private static final String KEY_PASSWORD = \"password\";\n    private static final String KEY_EMPTY = \"\";\n    private EditText etLogin,etPass;\n    private Button btnLogin;\n    private ProgressDialog pDialog;\n    public static String login_url = \"http://192.168.43.37/member/login.php\";\n    SessionHandler session;\n    private String username;\n    String id_user, password;\n    @Override\n    protected void onCreate(@Nullable Bundle savedInstanceState) {\n        super.onCreate(savedInstanceState);\n        setContentView(R.layout.activity_login);\n\n        etLogin = findViewById(R.id.etLoginUsername);\n        etPass = findViewById(R.id.etLoginPassword);\n        btnLogin = findViewById(R.id.btnLogin);\n        btnLogin.setOnClickListener(new View.OnClickListener() {\n            @Override\n            public void onClick(View v) {\n                username = etLogin.getText().toString().trim();\n                password = etPass.getText().toString().trim();\n\n                if (validateInputs()) {\n                    login();\n                }\n            }\n        });\n    }\n    private void displayLoader() {\n        pDialog = new ProgressDialog(LoginActivity.this);\n        pDialog.setMessage(\"Logging In.. Please wait...\");\n        pDialog.setIndeterminate(false);\n        pDialog.setCancelable(false);\n        pDialog.show();\n\n    }\n    private void login() {\n        displayLoader();\n        JSONObject request = new JSONObject();\n        try {\n            //Populate the request parameters\n            request.put(KEY_USERNAME, username);\n            request.put(KEY_PASSWORD, password);\n\n        } catch (JSONException e) {\n            e.printStackTrace();\n        }\n        JsonObjectRequest jsArrayRequest = new JsonObjectRequest\n                (Request.Method.POST, login_url, request, new Response.Listener<JSONObject>() {\n                    @Override\n                    public void onResponse(JSONObject response) {\n                        pDialog.dismiss();\n                        try {\n                            //Check if user got logged in successfully\n\n                            if (response.getInt(KEY_STATUS) == 0) {\n                                session.loginUser(username,response.getString(KEY_FULL_NAME));\n                                loadDashboard();\n\n                            }else{\n                                Toast.makeText(getApplicationContext(),\n                                        response.getString(KEY_MESSAGE), Toast.LENGTH_SHORT).show();\n\n                            }\n                        } catch (JSONException e) {\n                            e.printStackTrace();\n                        }\n                    }\n                }, new Response.ErrorListener() {\n\n                    @Override\n                    public void onErrorResponse(VolleyError error) {\n                        pDialog.dismiss();\n\n                        //Display error message whenever an error occurs\n                        Toast.makeText(getApplicationContext(),\n                                error.getMessage(), Toast.LENGTH_SHORT).show();\n\n                    }\n                });\n        RequestAntrian.getInstance(this).addToRequestQueue(jsArrayRequest);\n\n    }\n    private void loadDashboard()\n    {\n        Intent intent = new Intent(getApplicationContext(),Dashboard.class);\n        startActivity(intent);\n        finish();\n<｜fim▁hole｜>\n        }\n        if(KEY_EMPTY.equals(password)){\n            etPass.setError(\"Password cannot be empty\");\n            etPass.requestFocus();\n            return false;\n        }\n        return true;\n    }\n\n}\n<｜fim▁end｜>"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=2048)
print(tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_text):])