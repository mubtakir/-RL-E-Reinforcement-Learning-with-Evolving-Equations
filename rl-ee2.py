"""
RL-E²: نظام تعلم معزز مبتكر يعتمد على معادلات رياضية تتطوّر مع التدريب
تم تطويره بواسطة: [باسل يحيى عبدالله/ العراق/ الموصل ]
تاريخ الإصدار: [19/4/2025]
"""
# -*- coding: utf-8 -*-
"""
==============================================================================
نظام RL-E²: التعلم المعزز مع معادلات تطورية ديناميكية (نسخة محسنة v1.0 - DDPG-like)
==============================================================================

وصف النظام:
-------------
هذا البرنامج ينفذ نظام تعلم معزز مبتكر يُدعى RL-E² (Reinforcement Learning
with Evolving Equations). الفكرة الأساسية هي تمثيل سياسة الوكيل (Policy)
ودالة تقييم الفعل-الحالة (Q-Function / Critic) ليس باستخدام شبكات عصبونية تقليدية
ثابتة البنية، بل بواسطة معادلات رياضية مرنة تتطور بنيتها (عدد الحدود،
الدوال المستخدمة) ومعاملاتها (الأوزان، الأسس) ديناميكيًا أثناء عملية
التدريب. يهدف هذا النهج إلى تحقيق توازن أفضل بين القدرة التعبيرية للمعادلة
وقابليتها للتفسير، بالإضافة إلى التكيف الذاتي مع تعقيد المشكلة.

**التغييرات الرئيسية في v1.2:**
*   **الانتقال إلى بنية Actor-Critic شبيهة بـ DDPG:**
    *   تم استبدال دالة قيمة الحالة (V) بدالة قيمة الفعل-الحالة (Q).
    *   يستخدم الوكيل الآن شبكتين رئيسيتين: السياسة (Actor) والمقيّم (Critic - Q-function)، بالإضافة إلى شبكتي هدف (Target Networks) لكل منهما لزيادة استقرار التدريب.
    *   يتم تحديث المقيّم (Q-network) لتقليل خطأ TD باستخدام شبكات الهدف.
    *   يتم تحديث السياسة (Actor) لتعظيم قيمة Q المقدرة بواسطة المقيّم للإجراءات التي تتخذها السياسة.
*   **إصلاح خطأ التدرج:** تم إصلاح الخطأ الجوهري المتعلق بعدم تدفق التدرجات إلى شبكة السياسة عن طريق استخدام بنية Q-function و DDPG policy loss (`-Q(s, policy(s))`).
*   **تحديث شبكات الهدف:** تم إضافة آلية تحديث ناعم (Polyak averaging) لشبكات الهدف.
*   **تحديث المعادلات المتطورة:** تم تعديل `EvolvingEquation` لضمان `requires_grad=True` للمعاملات الجديدة عند الإضافة.
*   **تحديث محرك التطور:** تم تعديل `EvolutionEngine` ليعمل بشكل صحيح مع الوكيل الجديد (لا يزال يعيد تهيئة المحسن عند تغيير البنية). تم تعديل آلية تحديث شبكة الهدف المقابلة عند تغيير بنية الشبكة الرئيسية.
*   **تحسينات hyperparameters:** تم تعديل قيم الضوضاء ومعدلات التعلم المقترحة.
*   **توضيح التعليقات:** تم تحديث التعليقات لتعكس بنية DDPG-like الجديدة.

المكونات الرئيسية:
-----------------
1.  **EvolvingEquation (المعادلة المتطورة):**
    - تمثل معادلة رياضية مرنة.
    - **للـ Actor:** المدخلات هي الحالة (state)، المخرجات هي الإجراء الأولي (قبل tanh).
    - **للـ Critic (Q-function):** المدخلات هي الحالة والإجراء معًا (state + action)، المخرجات هي قيمة Q المقدرة (قيمة واحدة).
    - تحتفظ بقدرتها على التطور الهيكلي (إضافة/حذف حدود) وتطبيق آليات الاستقرار.
    - **تم التأكيد على `requires_grad=True` للمعاملات الجديدة.**

2.  **EvolutionEngine (محرك التطور):**
    - يدير تطور `EvolvingEquation` (إما Actor أو Critic).
    - يستخدم سجل الأداء لاتخاذ قرارات التطور.
    - **يعيد تهيئة محسن المعادلة المقابلة عند تغير بنيتها.** (يتطلب تمرير المحسن الصحيح).

3.  **ReplayBuffer (ذاكرة التجارب):**
    - ذاكرة تخزين قياسية (state, action, reward, next_state, done) مع فحص NaN.

4.  **RLE2Agent (وكيل RL-E² - DDPG-like):**
    - الوكيل الرئيسي المحدث.
    - يستخدم الآن 4 مثيلات من `EvolvingEquation`:
        - `policy_eq` (Actor)
        - `target_policy_eq` (Target Actor)
        - `q_eq` (Critic - Q-function)
        - `target_q_eq` (Target Critic)
    - يستخدم مثيلين من `EvolutionEngine` لإدارة تطور `policy_eq` و `q_eq`.
    - ينفذ خوارزمية شبيهة بـ DDPG:
        - تحديث المقيّم (Q-network) باستخدام خطأ TD: `Target = r + gamma * target_Q(s', target_policy(s')) * (1 - done)`.
        - تحديث السياسة (Actor) لتعظيم قيمة Q المقدرة: `Loss = -mean(Q(s, policy(s)))`.
        - تحديث شبكات الهدف باستخدام Polyak averaging (`tau`).
    - يطبق استكشافًا عبر ضوضاء غاوسية متناقصة *قبل* تطبيق `tanh`.
    - يدير عملية التحديث (أخذ عينات، حساب الخسائر، تحديث الشبكات، تحديث الهدف، التطور).
    - تم تحسين وظائف الحفظ والتحميل للتعامل مع الشبكات الأربع والمعادلات المتطورة.

5.  **train_rle2 (دالة التدريب الرئيسية):**
    - تنظم عملية التدريب باستخدام الوكيل المحدث.
    - تتفاعل مع بيئة Gym، تستخدم `RecordEpisodeStatistics`.
    - تدير حلقة التدريب، التقييم، الحفظ.
    - تستخدم `tqdm` لعرض التقدم.
    - تحفظ أفضل نموذج والنموذج النهائي.
    - **تم إضافة `matplotlib.use('Agg')` لتجنب مشاكل العرض في البيئات بدون واجهة رسومية عند حفظ الرسوم البيانية.**
    - تسمح باستئناف التدريب.

ميزات إضافية وتحسينات الاستقرار (المحافظ عليها وتلك المضافة في v1.2):
---------------------------------
- بنية معادلة مرنة مع طبقات خطية للمدخلات والمخرجات.
- تطور هيكلي يضبط الطبقات الخطية ويعيد تهيئة المحسن.
- معالجة NaN شاملة.
- عملية `pow` آمنة وتقييد القيم.
- تهيئة الأوزان (Xavier/Glorot).
- تطور مستقر يعتمد على نسبة الأداء.
- تحميل/حفظ قوي يعيد بناء المعادلات/المحسنات/الشبكات المستهدفة.
- توثيق شامل ومحدث.
- إدارة البذور العشوائية.
- رسم بياني محسن للنتائج.

"""

# --- 1. Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import random
from collections import deque
import matplotlib
matplotlib.use('Agg') # استخدام backend غير تفاعلي لتجنب أخطاء Qt/XCB في بيئات headless
import matplotlib.pyplot as plt
from tqdm import tqdm  # مكتبة لعرض شريط التقدم
import copy           # لنسخ الكائنات (مهم لشبكات الهدف)
import math           # للدوال الرياضية الأساسية
import time           # لتتبع وقت التدريب
import os             # لعمليات نظام الملفات (إنشاء مجلدات)
import warnings       # لإصدار التحذيرات

# تجاهل بعض التحذيرات غير الحرجة إذا لزم الأمر
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Converting SafeTensor to NumPy.*")


# --- 2. Global Configuration & Seed ---
RANDOM_SEED = 42
# تعيين البذرة العشوائية للمكتبات المختلفة لضمان قابلية تكرار التجارب
# (ملاحظة: بعض عمليات CUDA قد لا تكون قابلة للتكرار تمامًا)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # لـ multi-GPU
    # قد يؤدي تعطيل cuDNN benchmark إلى زيادة قابلية التكرار على حساب الأداء
    # torch.backends.cudnn.benchmark = False
    # قد يجعل العمليات الحتمية أبطأ
    # torch.backends.cudnn.deterministic = True
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --- 3. Core Component: Evolving Equation ---

class EvolvingEquation(nn.Module):
    """
    يمثل معادلة رياضية مرنة تتطور بنيتها ومعاملاتها ديناميكيًا.
    تُستخدم هذه الفئة لكل من السياسة (Actor) والمقيّم (Critic - Q-function).

    تتكون المعادلة من مجموع عدة "حدود" (terms). يتم حساب كل حد بناءً على:
    1.  ميزة مدخلة (feature): ناتجة عن تحويل خطي لمدخلات المعادلة الأصلية (الحالة للـ Actor، الحالة+الفعل للـ Critic).
    2.  أس (exponent): يتم رفع الميزة المدخلة إليه (مع معالجة آمنة).
    3.  دالة رياضية (function): يتم تطبيقها على نتيجة الرفع للأس (e.g., sin, tanh, relu).
    4.  معامل (coefficient): يتم ضرب نتيجة الدالة فيه.

    يتم دمج مخرجات جميع الحدود باستخدام طبقة خطية نهائية لإنتاج المخرج النهائي للمعادلة.
    يمكن للمعادلة إضافة حدود جديدة أو إزالة حدود قائمة أثناء عملية التطور، مع تعديل الطبقات المرتبطة.

    Attributes:
        input_dim (int): أبعاد المدخلات (state_dim للـ Actor, state_dim + action_dim للـ Critic).
        output_dim (int): أبعاد المخرجات (action_dim للـ Actor, 1 للـ Critic).
        complexity (int): العدد الحالي للحدود في المعادلة.
        complexity_limit (int): الحد الأقصى المسموح به لعدد الحدود.
        min_complexity (int): الحد الأدنى المسموح به لعدد الحدود.
        output_activation (callable, optional): دالة تنشيط تطبق على المخرج النهائي (None للـ Actor و Critic عادةً في DDPG).
        input_transform (nn.Linear): طبقة خطية لتحويل المدخلات إلى ميزات للحدود.
        coefficients (nn.ParameterList): قائمة بمعاملات كل حد (قابلة للتعلم).
        exponents (nn.ParameterList): قائمة بأسس كل حد (قابلة للتعلم).
        functions (list): قائمة بالدوال الرياضية المستخدمة حاليًا لكل حد.
        output_layer (nn.Linear): طبقة خطية لدمج مخرجات الحدود.
        function_library (list): قائمة بالدوال الرياضية المتاحة للاختيار منها.
        exp_clamp_min (float): الحد الأدنى لقيمة الأس المسموحة.
        exp_clamp_max (float): الحد الأقصى لقيمة الأس المسموحة.
        term_clamp (float): القيمة القصوى المطلقة المسموحة لمساهمة كل حد.
        output_clamp (float): القيمة القصوى المطلقة المسموحة للمخرج النهائي (قبل التنشيط).
    """
    def __init__(self, input_dim, init_complexity=3, output_dim=1,
                 complexity_limit=15, min_complexity=2, output_activation=None,
                 exp_clamp_min=0.1, exp_clamp_max=4.0, term_clamp=1e4, output_clamp=1e5):
        """
        تهيئة المعادلة المتطورة.

        Args:
            input_dim (int): أبعاد المدخلات. يجب أن يكون عددًا صحيحًا موجبًا.
            init_complexity (int): عدد الحدود الأولي. يجب أن يكون عددًا صحيحًا موجبًا.
            output_dim (int): أبعاد المخرجات. يجب أن يكون عددًا صحيحًا موجبًا.
            complexity_limit (int): الحد الأقصى لعدد الحدود. يجب أن يكون أكبر من أو يساوي min_complexity.
            min_complexity (int): الحد الأدنى لعدد الحدود. يجب أن يكون عددًا صحيحًا موجبًا.
            output_activation (callable, optional): دالة تنشيط اختيارية للمخرج. Defaults to None.
            exp_clamp_min (float): الحد الأدنى لقيمة الأس.
            exp_clamp_max (float): الحد الأقصى لقيمة الأس.
            term_clamp (float): القيمة القصوى المطلقة لكل حد.
            output_clamp (float): القيمة القصوى المطلقة للمخرج قبل التنشيط.
        """
        super().__init__()

        # --- التحقق من صحة المدخلات ---
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")
        if not isinstance(init_complexity, int) or init_complexity <= 0:
            raise ValueError("init_complexity must be a positive integer.")
        if not isinstance(min_complexity, int) or min_complexity <= 0:
            raise ValueError("min_complexity must be a positive integer.")
        if not isinstance(complexity_limit, int) or complexity_limit < min_complexity:
            raise ValueError(f"complexity_limit ({complexity_limit}) must be an integer >= min_complexity ({min_complexity}).")

        self.input_dim = input_dim
        self.output_dim = output_dim
        # ضمان أن التعقيد الأولي ضمن الحدود المسموحة
        self.complexity = max(min_complexity, min(init_complexity, complexity_limit))
        self.complexity_limit = complexity_limit
        self.min_complexity = min_complexity
        self.output_activation = output_activation
        self.exp_clamp_min = exp_clamp_min
        self.exp_clamp_max = exp_clamp_max
        self.term_clamp = term_clamp
        self.output_clamp = output_clamp

        # --- تعريف مكونات المعادلة ---

        # 1. طبقة تحويل المدخلات: (InputDim) -> (Complexity)
        self.input_transform = nn.Linear(input_dim, self.complexity)
        nn.init.xavier_uniform_(self.input_transform.weight, gain=nn.init.calculate_gain('relu')) # Xavier مناسب لمدخلات الطبقات التالية التي قد تستخدم ReLU أو مشابه
        nn.init.zeros_(self.input_transform.bias)

        # 2. مكونات الحدود (معاملات وأسس)
        self.coefficients = nn.ParameterList()
        self.exponents = nn.ParameterList()
        for _ in range(self.complexity):
            # التأكد من أن requires_grad=True (وهو الافتراضي لـ nn.Parameter لكن نجعله صريحًا)
            coeff = nn.Parameter(torch.randn(1) * 0.05, requires_grad=True)
            self.coefficients.append(coeff)
            exp = nn.Parameter(torch.abs(torch.randn(1) * 0.1) + 1.0, requires_grad=True)
            self.exponents.append(exp)


        # 3. مكتبة الدوال الرياضية المتاحة
        self.function_library = [
            torch.sin, torch.cos, torch.tanh, torch.sigmoid,
            F.relu, F.leaky_relu, F.gelu,
            lambda x: x, # Identity
            lambda x: torch.pow(x, 2), # Square
            # lambda x: torch.pow(x, 3), # Cube (يمكن أن يسبب عدم استقرار)
            lambda x: torch.exp(-torch.abs(x)), # Decaying Exp
            #lambda x: torch.log(torch.abs(x) + 1e-6), # Safe Log Abs (قد يكون غير مستقر)
            lambda x: torch.sqrt(torch.abs(x) + 1e-6), # Safe Sqrt Abs
            lambda x: torch.clamp(x, -3.0, 3.0), # Clamp (نطاق أضيق قد يساعد)
            lambda x: x * torch.sigmoid(x), # Swish/SiLU variant
        ]
        if not self.function_library:
            raise ValueError("Function library cannot be empty.")

        # 4. اختيار الدوال الأولية
        self.functions = [self._select_function() for _ in range(self.complexity)]

        # 5. طبقة الدمج النهائية: (Complexity) -> (OutputDim)
        self.output_layer = nn.Linear(self.complexity, output_dim)
        # تهيئة مناسبة للمخرجات الخطية (قد تكون سالبة أو موجبة)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.zeros_(self.output_layer.bias)

    def _select_function(self):
        """يختار دالة رياضية عشوائية من مكتبة الدوال المتاحة."""
        return random.choice(self.function_library)

    def _safe_pow(self, base, exp):
        """
        عملية رفع للأس آمنة لتجنب NaN من pow(negative, fractional) أو pow(0, non_positive).
        """
        # التأكد من أن الأس موجب (يتم تقييده خارج الدالة بالفعل)
        # التعامل مع الأساس
        sign = torch.sign(base)
        # استخدام القيمة المطلقة + epsilon صغير جدًا
        base_abs_safe = torch.abs(base) + 1e-8
        powered = torch.pow(base_abs_safe, exp)

        # إعادة الإشارة الأصلية. ملاحظة: هذا ليس مكافئًا تمامًا لـ x^n رياضيًا
        # خاصة للأسس الكسرية، ولكنه يحافظ على الاستقرار العددي.
        # طريقة بديلة: إعادة الإشارة فقط إذا كان الأس فرديًا (أكثر تعقيدًا)
        return sign * powered

    def forward(self, x):
        """
        ينفذ التمرير الأمامي للمعادلة، حاسبًا المخرج بناءً على المدخلات الحالية.
        يتضمن آليات لزيادة الاستقرار العددي.
        """
        # --- 0. التحقق الأولي وإعداد المدخلات ---
        if not isinstance(x, torch.Tensor):
             try:
                 x = torch.tensor(x, dtype=torch.float32)
             except Exception as e:
                 raise TypeError(f"Input 'x' must be a tensor or convertible to a tensor. Error: {e}")

        if x.dim() == 1:
            # إذا كان البعد 1، نفترض أنها عينة واحدة ونضيف بعد الدفعة
            # ونضمن أنها تطابق input_dim
            if x.shape[0] == self.input_dim:
                 x = x.unsqueeze(0)
            else:
                 raise ValueError(f"Input dimension mismatch for single sample. Expected {self.input_dim}, got {x.shape[0]}")
        elif x.dim() > 2:
             # تسوية المدخلات إذا كانت ذات أبعاد أعلى (مثل الصور)
             # ملاحظة: قد لا يكون هذا مناسبًا لجميع الحالات، لكنه يعالج الخطأ
             original_shape = x.shape
             x = x.view(original_shape[0], -1)
             warnings.warn(f"Input tensor has more than 2 dimensions ({original_shape}). Flattening input to ({x.shape}).", RuntimeWarning)


        if x.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch. Expected {self.input_dim}, got {x.shape[1]}")

        if torch.isnan(x).any():
            warnings.warn(f"NaN detected in input 'x' to EvolvingEquation (Shape: {x.shape}). Replacing with 0.", RuntimeWarning)
            x = torch.nan_to_num(x, nan=0.0)

        # نقل x إلى نفس جهاز النموذج إذا لزم الأمر
        model_device = next(self.parameters()).device
        if x.device != model_device:
            x = x.to(model_device)

        # --- 1. تحويل المدخلات إلى ميزات وسيطة ---
        try:
            # transformed_features: (BatchSize, Complexity)
            transformed_features = self.input_transform(x)
            if torch.isnan(transformed_features).any():
                warnings.warn("NaN detected in 'transformed_features' after input_transform. Replacing with 0.", RuntimeWarning)
                transformed_features = torch.nan_to_num(transformed_features, nan=0.0)
            # تقييد الميزات قبل العمليات غير الخطية قد يساعد
            transformed_features = torch.clamp(transformed_features, -self.term_clamp, self.term_clamp)

        except Exception as e:
            warnings.warn(f"Exception during input_transform: {e}. Returning zeros.", RuntimeWarning)
            return torch.zeros(x.shape[0], self.output_dim, device=x.device)

        # --- 2. حساب مساهمة كل حد في المعادلة ---
        term_results = torch.zeros(x.shape[0], self.complexity, device=x.device)
        for i in range(self.complexity):
            try:
                # feature_i: (BatchSize,)
                feature_i = transformed_features[:, i]

                # تقييد الأس ضمن نطاق معقول
                exp_val = torch.clamp(self.exponents[i], self.exp_clamp_min, self.exp_clamp_max)

                # --- المعالجة الآمنة لعملية الرفع للأس (pow) ---
                term_powered = self._safe_pow(feature_i, exp_val)

                if torch.isnan(term_powered).any() or torch.isinf(term_powered).any():
                    warnings.warn(f"NaN/Inf detected after safe pow in term {i}. Feature mean: {feature_i.mean().item():.2f}, Exp: {exp_val.item():.2f}. Replacing with 0.", RuntimeWarning)
                    term_powered = torch.zeros_like(feature_i)
                term_powered = torch.clamp(term_powered, -self.term_clamp, self.term_clamp) # تقييد بعد الرفع

                # --- تطبيق الدالة الرياضية ---
                term_activated = self.functions[i](term_powered)

                if torch.isnan(term_activated).any() or torch.isinf(term_activated).any():
                    func_name = self.functions[i].__name__ if hasattr(self.functions[i], '__name__') else 'lambda'
                    warnings.warn(f"NaN/Inf detected after applying function '{func_name}' in term {i}. Input mean: {term_powered.mean().item():.2f}. Replacing with 0.", RuntimeWarning)
                    term_activated = torch.zeros_like(feature_i)
                term_activated = torch.clamp(term_activated, -self.term_clamp, self.term_clamp) # تقييد بعد الدالة

                # --- ضرب في المعامل وتقييد قيمة الحد ---
                term_value = self.coefficients[i] * term_activated
                # تقييد قيمة الحد الفردي
                term_results[:, i] = torch.clamp(term_value, -self.term_clamp, self.term_clamp)

            except Exception as e:
                func_name = self.functions[i].__name__ if hasattr(self.functions[i], '__name__') else 'lambda'
                warnings.warn(f"Exception during term {i} (func: {func_name}) calculation: {e}. Setting term result to 0.", RuntimeWarning)
                term_results[:, i] = 0.0

        # --- 3. دمج نتائج الحدود وتطبيق التنشيط النهائي ---
        if torch.isnan(term_results).any() or torch.isinf(term_results).any():
            warnings.warn("NaN/Inf detected in 'term_results' before output layer. Replacing with 0.", RuntimeWarning)
            term_results = torch.nan_to_num(term_results, nan=0.0, posinf=self.term_clamp, neginf=-self.term_clamp)

        try:
            output = self.output_layer(term_results)

            # تقييد المخرج قبل التنشيط
            output = torch.clamp(output, -self.output_clamp, self.output_clamp)

            if torch.isnan(output).any() or torch.isinf(output).any():
                warnings.warn("NaN/Inf detected after output layer (pre-activation). Replacing with safe values.", RuntimeWarning)
                # استبدال بقيم آمنة (مثل 0 أو قيمة مقيدة)
                output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

            # تطبيق دالة التنشيط النهائية إذا تم توفيرها
            if self.output_activation:
                output = self.output_activation(output)
                if torch.isnan(output).any() or torch.isinf(output).any():
                    warnings.warn("NaN/Inf detected after final activation. Replacing with safe values.", RuntimeWarning)
                    output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0) # افتراض النطاق [-1, 1] للتنشيطات الشائعة

        except Exception as e:
            warnings.warn(f"Exception during output layer or activation: {e}. Returning zeros.", RuntimeWarning)
            return torch.zeros(x.shape[0], self.output_dim, device=x.device)

        # --- 4. فحص المخرج النهائي ---
        if torch.isnan(output).any() or torch.isinf(output).any():
             warnings.warn(f"NaN/Inf detected in the FINAL output of EvolvingEquation (Shape: {output.shape}). Returning zeros.", RuntimeWarning)
             output = torch.zeros_like(output)


        return output

    def add_term(self):
        """
        يضيف حدًا جديدًا للمعادلة إذا كان التعقيد الحالي أقل من الحد الأقصى.
        يعدل طبقات `input_transform` و `output_layer` بشكل صحيح.
        يضمن أن المعلمات الجديدة تتطلب تدرجًا.

        Returns:
            bool: True إذا تمت الإضافة بنجاح، False خلاف ذلك.

        ملاحظة: يجب إعادة تهيئة المحسن بعد استدعاء هذه الدالة.
               يجب تحديث شبكة الهدف المقابلة (إذا كانت موجودة) بعد استدعاء هذه الدالة.
        """
        if self.complexity >= self.complexity_limit:
            return False

        new_complexity = self.complexity + 1
        device = next(self.parameters()).device # الحصول على الجهاز الحالي

        # --- حفظ الأوزان القديمة ---
        try:
            old_input_weight = self.input_transform.weight.data.clone()
            old_input_bias = self.input_transform.bias.data.clone()
            old_output_weight = self.output_layer.weight.data.clone()
            old_output_bias = self.output_layer.bias.data.clone()
        except AttributeError:
            warnings.warn("Could not access old weights during add_term. This might happen if layers are None.", RuntimeWarning)
            return False # لا يمكن المتابعة بأمان

        # --- إضافة مكونات الحد الجديد ---
        # تهيئة صغيرة للمعامل والأس الجديد مع التأكيد على requires_grad=True
        new_coeff = nn.Parameter(torch.randn(1, device=device) * 0.01, requires_grad=True)
        self.coefficients.append(new_coeff)
        # تهيئة الأس ليكون قريبًا من 1 وموجبًا
        new_exp = nn.Parameter(torch.abs(torch.randn(1, device=device) * 0.05) + 1.0, requires_grad=True)
        self.exponents.append(new_exp)
        self.functions.append(self._select_function())

        # --- توسيع طبقة المدخلات ---
        new_input_transform = nn.Linear(self.input_dim, new_complexity, device=device)
        with torch.no_grad():
            new_input_transform.weight.data[:self.complexity, :] = old_input_weight
            new_input_transform.bias.data[:self.complexity] = old_input_bias
            # تهيئة الجزء الجديد من الأوزان والانحياز
            nn.init.xavier_uniform_(new_input_transform.weight.data[self.complexity:], gain=nn.init.calculate_gain('relu'))
            new_input_transform.weight.data[self.complexity:] *= 0.01 # تهيئة صغيرة
            nn.init.zeros_(new_input_transform.bias.data[self.complexity:])
        self.input_transform = new_input_transform

        # --- توسيع طبقة المخرجات ---
        new_output_layer = nn.Linear(new_complexity, self.output_dim, device=device)
        with torch.no_grad():
            new_output_layer.weight.data[:, :self.complexity] = old_output_weight
            # تهيئة الجزء الجديد من أوزان المخرجات
            nn.init.xavier_uniform_(new_output_layer.weight.data[:, self.complexity:], gain=nn.init.calculate_gain('linear'))
            new_output_layer.weight.data[:, self.complexity:] *= 0.01 # تهيئة صغيرة
            # الانحياز القديم يبقى كما هو
            new_output_layer.bias.data.copy_(old_output_bias)
        self.output_layer = new_output_layer

        self.complexity = new_complexity
        # print(f"DEBUG: Term added. New complexity: {self.complexity}")
        return True

    def prune_term(self, aggressive=False):
        """
        يزيل حدًا من المعادلة إذا كان التعقيد الحالي أكبر من الحد الأدنى.
        يعدل طبقات `input_transform` و `output_layer` بشكل صحيح.

        Args:
            aggressive (bool): إذا كانت True، يحاول إزالة الحد الأقل أهمية.
                               إذا كانت False، يزيل حدًا عشوائيًا.

        Returns:
            bool: True إذا تم الحذف بنجاح، False خلاف ذلك.

        ملاحظة: يجب إعادة تهيئة المحسن بعد استدعاء هذه الدالة.
               يجب تحديث شبكة الهدف المقابلة (إذا كانت موجودة) بعد استدعاء هذه الدالة.
        """
        if self.complexity <= self.min_complexity:
            return False

        new_complexity = self.complexity - 1
        device = next(self.parameters()).device # الحصول على الجهاز الحالي

        # --- اختيار الفهرس للحذف ---
        idx_to_prune = -1
        try:
            if aggressive and self.complexity > 1: # لا يمكن حساب الأهمية لحد واحد
                 with torch.no_grad():
                    # حساب الأهمية: abs(coeff) * (L1_norm(input_weight_row) + L1_norm(output_weight_col))
                    # التأكد من أن جميع المكونات على نفس الجهاز (CPU للحساب)
                    coeffs_abs = torch.tensor([torch.abs(c.data).item() for c in self.coefficients], device='cpu')
                    input_weights_norm = torch.norm(self.input_transform.weight.data.cpu(), p=1, dim=1) if self.input_transform.weight is not None else torch.zeros(self.complexity, device='cpu')
                    output_weights_norm = torch.norm(self.output_layer.weight.data.cpu(), p=1, dim=0) if self.output_layer.weight is not None else torch.zeros(self.complexity, device='cpu')

                    if coeffs_abs.shape[0] == self.complexity and \
                       input_weights_norm.shape[0] == self.complexity and \
                       output_weights_norm.shape[0] == self.complexity:
                        # إضافة epsilon صغير
                        importance = (coeffs_abs * (input_weights_norm + output_weights_norm)) + 1e-9
                        # التحقق من عدم وجود NaN في الأهمية
                        if torch.isnan(importance).any():
                             warnings.warn("NaN detected in importance calculation during aggressive prune. Pruning randomly.", RuntimeWarning)
                             idx_to_prune = random.randint(0, self.complexity - 1)
                        else:
                            idx_to_prune = torch.argmin(importance).item()
                            # print(f"DEBUG: Aggressive prune selected index {idx_to_prune} with importance {importance[idx_to_prune]:.4g}")
                    else:
                        warnings.warn(f"Dimension mismatch during aggressive prune importance calc (Coeffs:{coeffs_abs.shape}, InNorm:{input_weights_norm.shape}, OutNorm:{output_weights_norm.shape}, Comp:{self.complexity}). Pruning randomly.", RuntimeWarning)
                        idx_to_prune = random.randint(0, self.complexity - 1)
            else:
                idx_to_prune = random.randint(0, self.complexity - 1)
                # print(f"DEBUG: Random prune selected index {idx_to_prune}")

            if not (0 <= idx_to_prune < self.complexity):
                 warnings.warn(f"Invalid index {idx_to_prune} selected for pruning (Complexity: {self.complexity}). Skipping prune.", RuntimeWarning)
                 return False

        except Exception as e:
            warnings.warn(f"Exception during index selection for pruning: {e}. Skipping prune.", RuntimeWarning)
            return False

        # --- حفظ الأوزان القديمة ---
        try:
            old_input_weight = self.input_transform.weight.data.clone()
            old_input_bias = self.input_transform.bias.data.clone()
            old_output_weight = self.output_layer.weight.data.clone()
            old_output_bias = self.output_layer.bias.data.clone()
        except AttributeError:
            warnings.warn("Could not access old weights during prune_term.", RuntimeWarning)
            return False

        # --- إزالة المكونات ---
        try:
            # يجب حذف العناصر من ParameterList قبل تغيير الطبقات
            del self.coefficients[idx_to_prune]
            del self.exponents[idx_to_prune]
            self.functions.pop(idx_to_prune)
        except (IndexError, Exception) as e:
            warnings.warn(f"Failed to remove components at index {idx_to_prune}: {e}. Aborting prune.", RuntimeWarning)
            # إعادة إضافة المعاملات المحذوفة إذا فشلت العملية لاحقًا؟ معقد. الأفضل الإيقاف.
            return False

        # --- تصغير طبقة المدخلات ---
        new_input_transform = nn.Linear(self.input_dim, new_complexity, device=device)
        with torch.no_grad():
            new_weight_in = torch.cat([old_input_weight[:idx_to_prune], old_input_weight[idx_to_prune+1:]], dim=0)
            new_bias_in = torch.cat([old_input_bias[:idx_to_prune], old_input_bias[idx_to_prune+1:]])
            if new_weight_in.shape[0] != new_complexity or new_bias_in.shape[0] != new_complexity:
                 warnings.warn(f"Dimension mismatch after input transform prune calculation. Aborting prune.", RuntimeWarning)
                 # محاولة استعادة الحالة قبل الحذف؟ صعب.
                 return False
            new_input_transform.weight.data.copy_(new_weight_in)
            new_input_transform.bias.data.copy_(new_bias_in)
        self.input_transform = new_input_transform

        # --- تصغير طبقة المخرجات ---
        new_output_layer = nn.Linear(new_complexity, self.output_dim, device=device)
        with torch.no_grad():
            new_weight_out = torch.cat([old_output_weight[:, :idx_to_prune], old_output_weight[:, idx_to_prune+1:]], dim=1)
            if new_weight_out.shape[1] != new_complexity:
                 warnings.warn(f"Dimension mismatch after output layer prune calculation. Aborting prune.", RuntimeWarning)
                 return False
            new_output_layer.weight.data.copy_(new_weight_out)
            # الانحياز لا يتأثر بحذف ميزة دخل
            new_output_layer.bias.data.copy_(old_output_bias)
        self.output_layer = new_output_layer

        self.complexity = new_complexity
        # print(f"DEBUG: Term pruned. New complexity: {self.complexity}")
        return True

# --- 4. Core Component: Evolution Engine ---

class EvolutionEngine:
    """
    محرك التطور: يدير عملية تطور كائن `EvolvingEquation`.
    يستخدم سجل الأداء لاتخاذ قرارات تطورية، ويطبق فترة تبريد،
    ويعيد تهيئة المحسن عند تغيير البنية.
    """
    def __init__(self, mutation_power=0.03, history_size=50, cooldown_period=30,
                 add_term_threshold=0.85, prune_term_threshold=0.20,
                 add_term_prob=0.15, prune_term_prob=0.25, swap_func_prob=0.02):
        """
        تهيئة محرك التطور.

        Args:
            mutation_power (float): قوة التحور الأساسية (e.g., 0.01-0.05).
            history_size (int): حجم نافذة سجل الأداء.
            cooldown_period (int): عدد **خطوات التحديث** للانتظار بعد تغيير البنية.
            add_term_threshold (float): العتبة المئوية للأداء للنظر في إضافة حد (e.g., 0.85). الأداء الجيد يشجع على الإضافة لاستكشاف تعقيد أكبر.
            prune_term_threshold (float): العتبة المئوية للأداء للنظر في حذف حد (e.g., 0.20). الأداء السيء يشجع على الحذف لتبسيط المعادلة.
            add_term_prob (float): الاحتمالية الفعلية للإضافة إذا تم تجاوز العتبة.
            prune_term_prob (float): الاحتمالية الفعلية للحذف إذا تم تجاوز العتبة.
            swap_func_prob (float): الاحتمالية لتبديل دالة في كل خطوة تطور.
        """
        # إضافة بعض التحققات الأساسية للمعلمات
        if not (0 < mutation_power < 0.5): warnings.warn(f"mutation_power ({mutation_power}) is outside the typical (0, 0.5) range.")
        if not (0 < add_term_threshold < 1): warnings.warn(f"add_term_threshold ({add_term_threshold}) is outside (0, 1).")
        if not (0 < prune_term_threshold < 1): warnings.warn(f"prune_term_threshold ({prune_term_threshold}) is outside (0, 1).")
        if add_term_threshold <= prune_term_threshold: warnings.warn(f"add_term_threshold <= prune_term_threshold. This might hinder growth.")
        if not (0 <= add_term_prob <= 1) or not (0 <= prune_term_prob <= 1) or not (0 <= swap_func_prob <= 1): warnings.warn("Evolution probabilities should be between 0 and 1.")


        self.base_mutation_power = mutation_power
        self.performance_history = deque(maxlen=history_size)
        self.cooldown_period = cooldown_period
        self.term_change_cooldown = 0 # عداد فترة الانتظار

        self.add_term_threshold = add_term_threshold
        self.prune_term_threshold = prune_term_threshold
        self.add_term_prob = add_term_prob
        self.prune_term_prob = prune_term_prob
        self.swap_func_prob = swap_func_prob

    def _calculate_percentile(self, current_reward):
        """
        يحسب النسبة المئوية للأداء الحالي مقارنة بسجل الأداء التاريخي (مع معالجة NaN).
        النسبة المئوية هنا تمثل "كم نسبة من الأداء التاريخي أسوأ من الأداء الحالي".
        أداء عالٍ -> نسبة مئوية عالية.
        """
        if math.isnan(current_reward):
            warnings.warn("NaN reward passed to _calculate_percentile. Returning 0.5 (neutral).", RuntimeWarning)
            return 0.5

        valid_history = [r for r in self.performance_history if not math.isnan(r)]
        if not valid_history:
             # لا يوجد سجل كافٍ، افترض أداء متوسط
             return 0.5

        # استخدام numpy للتعامل الفعال مع الحسابات
        history_array = np.array(valid_history)
        # حساب النسبة المئوية كنسبة القيم الأقل من القيمة الحالية
        # kind='mean' يعطي تقديرًا أكثر سلاسة إذا تكررت القيم
        from scipy import stats
        try:
             # استخدام scipy.stats.percentileofscore لحساب أكثر دقة
             percentile = stats.percentileofscore(history_array, current_reward, kind='mean') / 100.0
        except Exception as e:
             warnings.warn(f"Scipy percentile calculation failed: {e}. Using numpy mean comparison.", RuntimeWarning)
             percentile = np.mean(history_array < current_reward)

        # التأكد من أن النسبة المئوية بين 0 و 1
        return np.clip(percentile, 0.0, 1.0)


    def _dynamic_mutation_scale(self, percentile):
        """
        يحدد عامل قياس لقوة التحور بناءً على النسبة المئوية للأداء.
        (تقليل عند الأداء العالي جدًا، زيادة عند المنخفض جدًا)
        يهدف إلى زيادة الاستكشاف عند الأداء السيء وتقليل الإزعاج عند الأداء الجيد.
        """
        if math.isnan(percentile):
             warnings.warn("NaN percentile passed to _dynamic_mutation_scale. Using scale 1.0.", RuntimeWarning)
             percentile = 0.5

        # مثال: مقياس يتراوح بين 0.5 (للأداء الممتاز > 90%) و 1.5 (للأداء الضعيف < 10%)
        if percentile > 0.9:
            scale = 0.5
        elif percentile < 0.1:
            scale = 1.5
        else:
            # تدرج خطي في المنتصف
            scale = 1.5 - (percentile - 0.1) * (1.5 - 0.5) / (0.9 - 0.1)

        return max(0.1, min(scale, 2.0)) # تقييد المقياس ضمن نطاق معقول


    def evolve(self, equation, reward, step, optimizer):
        """
        ينفذ خطوة تطور واحدة للمعادلة، يعيد تهيئة المحسن إذا لزم الأمر.

        Args:
            equation (EvolvingEquation): المعادلة المراد تطويرها.
            reward (float): المكافأة المتوسطة الحديثة (مؤشر الأداء).
            step (int): رقم خطوة **التحديث** الحالية في التدريب (للتبريد).
            optimizer (torch.optim.Optimizer): المحسن الحالي للمعادلة.

        Returns:
            tuple: (torch.optim.Optimizer, bool)
                - المحسن (قد يكون مثيلًا جديدًا إذا تغيرت البنية).
                - علامة (bool) تشير إلى ما إذا كانت بنية المعادلة قد تغيرت.
        """
        structure_changed = False
        original_optimizer = optimizer
        current_lr = optimizer.param_groups[0]['lr']
        weight_decay = optimizer.param_groups[0].get('weight_decay', 0)

        # --- 1. تسجيل الأداء وحساب النسبة المئوية ---
        if not math.isnan(reward):
            self.performance_history.append(reward)
            percentile = self._calculate_percentile(reward)
        else:
            # إذا كانت المكافأة NaN، استخدم آخر نسبة مئوية صالحة أو قيمة محايدة
            warnings.warn(f"NaN reward received at update step {step}. Using last valid percentile or 0.5 for evolution.", RuntimeWarning)
            if self.performance_history:
                valid_history = [r for r in self.performance_history if not math.isnan(r)]
                if valid_history:
                     percentile = self._calculate_percentile(valid_history[-1])
                else:
                     percentile = 0.5 # لا توجد مكافآت صالحة سابقة
            else:
                 percentile = 0.5 # لا يوجد سجل أداء على الإطلاق

        # --- 2. تقليل فترة الانتظار (Cooldown) ---
        if self.term_change_cooldown > 0:
            self.term_change_cooldown -= 1

        # --- 3. قرار تغيير البنية (إضافة/حذف) ---
        # يتطلب وجود سجل أداء كافٍ لاتخاذ قرار مستنير
        if self.term_change_cooldown == 0 and len(self.performance_history) >= max(10, self.performance_history.maxlen // 4):
            rand_roll = random.random()
            action_taken = "None"

            # محاولة إضافة حد (إذا كان الأداء جيدًا جدًا)
            if percentile > self.add_term_threshold and rand_roll < self.add_term_prob:
                if equation.add_term():
                    self.term_change_cooldown = self.cooldown_period
                    structure_changed = True
                    action_taken = f"Added Term (New Comp: {equation.complexity})"

            # محاولة حذف حد (إذا كان الأداء سيئًا جدًا)
            # استخدام الحذف العدواني (إزالة الأقل أهمية)
            elif percentile < self.prune_term_threshold and rand_roll < self.prune_term_prob:
                if equation.prune_term(aggressive=True):
                    self.term_change_cooldown = self.cooldown_period
                    structure_changed = True
                    action_taken = f"Pruned Term (New Comp: {equation.complexity})"

            if structure_changed:
                 # استخدم print بدلاً من tqdm.write هنا لأن evolve يُستدعى داخل update، وقد لا يكون tqdm متاحًا مباشرة
                 print(f"\nINFO [{time.strftime('%H:%M:%S')}]: Update Step {step}: Equation '{type(equation).__name__}' Evolution: {action_taken} | Perf Percentile: {percentile:.2f} | Cooldown: {self.term_change_cooldown}")


        # --- 4. قرار تبديل دالة ---
        # يتم فقط إذا لم يتغير الهيكل وفي حالة عدم وجود تبريد
        if not structure_changed and self.term_change_cooldown == 0 and random.random() < self.swap_func_prob:
            if self.swap_function(equation):
                 # فترة تبريد قصيرة جدًا بعد تبديل الدالة للسماح للأوزان بالتكيف قليلاً
                 self.term_change_cooldown = max(self.term_change_cooldown, 2) # تبريد قصير جدًا
                 # print(f"DEBUG [{time.strftime('%H:%M:%S')}]: Step {step}: Swapped function in equation.")


        # --- 5. تطبيق التحور (Mutation) للمعاملات ---
        # التحور يطبق دائمًا (إلا إذا كان المقياس صغيرًا جدًا)
        mutation_scale = self._dynamic_mutation_scale(percentile)
        self._mutate_parameters(equation, mutation_scale, step)

        # --- 6. إعادة تهيئة المحسن إذا تغيرت بنية المعادلة ---
        if structure_changed:
            # print(f"DEBUG [{time.strftime('%H:%M:%S')}]: Step {step}: Reinitializing optimizer due to structure change.")
            try:
                optim_class = type(optimizer)
                # الحصول على المعلمات الجديدة (يجب أن تكون موجودة بعد التغيير)
                new_params = list(equation.parameters())

                if not new_params:
                    warnings.warn(f"Equation has no parameters after structure change at step {step}! Cannot reinitialize optimizer.", RuntimeWarning)
                    optimizer = original_optimizer # إعادة المحسن الأصلي (قد يكون غير صالح)
                    structure_changed = False # التراجع عن اعتبار الهيكل متغيرًا إذا لم نتمكن من إنشاء محسن جديد
                else:
                    # إنشاء محسن جديد بنفس الإعدادات الرئيسية (LR, WD)
                    # لا يمكن الحفاظ على حالة المحسن (مثل العزوم في Adam) بسهولة عند تغيير مجموعة المعلمات
                    optimizer = optim_class(new_params, lr=current_lr, weight_decay=weight_decay)
                    # print(f"DEBUG [{time.strftime('%H:%M:%S')}]: Optimizer reinitialized successfully for {len(new_params)} parameters.")

            except Exception as e:
                 warnings.warn(f"Failed to reinitialize optimizer at step {step}: {e}. Returning original.", RuntimeWarning)
                 optimizer = original_optimizer # محاولة إرجاع المحسن الأصلي
                 structure_changed = False # التراجع

        return optimizer, structure_changed

    def _mutate_parameters(self, equation, mutation_scale, step):
        """
        يطبق تحورًا (ضوضاء غاوسية) على معلمات المعادلة القابلة للتعلم.
        التحور الآن يؤثر على المعاملات، الأسس، وطبقات التحويل/المخرجات.
        """
        # عامل تبريد يقلل قوة التحور الكلية مع مرور الوقت بشكل طفيف جدًا
        # (يصل إلى 50% من القوة الأصلية بعد 2 مليون خطوة تحديث)
        cooling_factor = max(0.5, 1.0 - step / 2000000.0)
        effective_power = self.base_mutation_power * mutation_scale * cooling_factor

        if effective_power < 1e-9: # تجنب التحور الضئيل جدًا
             return

        try:
            with torch.no_grad():
                # التأكد من وجود معاملات قبل محاولة الوصول إليها
                if not list(equation.parameters()):
                    # print(f"DEBUG [{time.strftime('%H:%M:%S')}]: Equation has no parameters to mutate at step {step}.")
                    return

                device = next(equation.parameters()).device

                # تحور المعاملات (Coefficients)
                if hasattr(equation, 'coefficients') and equation.coefficients:
                    for coeff in equation.coefficients:
                        noise = torch.randn_like(coeff.data) * effective_power
                        coeff.data.add_(noise)
                        # coeff.data.clamp_(-10.0, 10.0) # تقييد اختياري للمعاملات

                # تحور الأسس (Exponents) بقوة أقل وتقييدها
                if hasattr(equation, 'exponents') and equation.exponents:
                    for exp in equation.exponents:
                        noise = torch.randn_like(exp.data) * effective_power * 0.1 # قوة أقل للأسس
                        exp.data.add_(noise)
                        exp.data.clamp_(min=equation.exp_clamp_min, max=equation.exp_clamp_max) # تطبيق التقييد

                # تحور طبقة المدخلات (input_transform)
                if hasattr(equation, 'input_transform') and isinstance(equation.input_transform, nn.Linear):
                    power_scale = 0.3 # قوة أقل لطبقات التحويل
                    noise_w_in = torch.randn_like(equation.input_transform.weight.data) * effective_power * power_scale
                    equation.input_transform.weight.data.add_(noise_w_in)
                    if equation.input_transform.bias is not None:
                        noise_b_in = torch.randn_like(equation.input_transform.bias.data) * effective_power * power_scale * 0.5 # انحياز بقوة أقل
                        equation.input_transform.bias.data.add_(noise_b_in)

                # تحور طبقة المخرجات (output_layer)
                if hasattr(equation, 'output_layer') and isinstance(equation.output_layer, nn.Linear):
                    power_scale = 0.3 # قوة أقل لطبقات التحويل
                    noise_w_out = torch.randn_like(equation.output_layer.weight.data) * effective_power * power_scale
                    equation.output_layer.weight.data.add_(noise_w_out)
                    if equation.output_layer.bias is not None:
                        noise_b_out = torch.randn_like(equation.output_layer.bias.data) * effective_power * power_scale * 0.5 # انحياز بقوة أقل
                        equation.output_layer.bias.data.add_(noise_b_out)

        except StopIteration:
             # يحدث إذا كانت قائمة المعلمات فارغة (تم التعامل معه في البداية)
            pass
        except Exception as e:
            warnings.warn(f"Exception during parameter mutation at step {step}: {e}", RuntimeWarning)

    def swap_function(self, equation):
        """
        يبدل دالة حد عشوائي بأخرى من المكتبة (يحاول اختيار دالة مختلفة).
        """
        if equation.complexity <= 0:
            return False

        try:
            idx_to_swap = random.randint(0, equation.complexity - 1)
            old_func = equation.functions[idx_to_swap]
            old_func_repr = old_func.__name__ if hasattr(old_func, '__name__') else repr(old_func)

            attempts = 0
            max_attempts = len(equation.function_library) * 2
            new_func = old_func # قيمة أولية

            # محاولة العثور على دالة *مختلفة*
            while attempts < max_attempts:
                candidate_func = equation._select_function()
                # المقارنة بالمرجع قد لا تكون كافية (خاصة لـ lambda). يمكن مقارنة الأسماء.
                candidate_repr = candidate_func.__name__ if hasattr(candidate_func, '__name__') else repr(candidate_func)
                # نعتبر lambda مختلفة عن بعضها البعض إذا كانت مراجعها مختلفة
                is_different = (candidate_repr != old_func_repr) or \
                               (candidate_repr == 'lambda' and candidate_func is not old_func) or \
                               (len(equation.function_library) == 1) # إذا كانت هناك دالة واحدة فقط، فالاختيار دائمًا هو نفسه

                if is_different or attempts == max_attempts - 1: # إذا فشلنا في العثور على مختلف، نستخدم آخر محاولة
                    new_func = candidate_func
                    break
                attempts += 1

            # تطبيق التغيير
            equation.functions[idx_to_swap] = new_func
            new_func_repr = new_func.__name__ if hasattr(new_func, '__name__') else repr(new_func)
            if new_func is not old_func:
                 # print(f"DEBUG: Swapped function at index {idx_to_swap} from '{old_func_repr}' to '{new_func_repr}'.")
                 pass
            return True

        except (IndexError, Exception) as e:
            warnings.warn(f"Exception during function swap: {e}", RuntimeWarning)
            return False


# --- 5. Core Component: Replay Buffer ---

class ReplayBuffer:
    """
    ذاكرة تخزين مؤقت للتجارب مع فحص NaN عند الإضافة وأخذ العينات.
    """
    def __init__(self, capacity=100000):
        if not isinstance(capacity, int) or capacity <= 0:
             raise ValueError("Replay buffer capacity must be a positive integer.")
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self._push_nan_warnings = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}
        self._sample_nan_warning = 0

    def push(self, state, action, reward, next_state, done):
        """يضيف تجربة مع فحص NaN وتحويل الأنواع."""
        skip_experience = False
        nan_source = None

        # --- فحص NaN قبل الإضافة ---
        try:
            # التحقق من المكافأة
            if isinstance(reward, (int, float)) and (math.isnan(reward) or math.isinf(reward)):
                # تسجيل المكافأة كـ 0 بدلاً من تخطي التجربة كلها
                reward = 0.0
                self._push_nan_warnings['reward'] += 1
                nan_source = 'reward'
                # لا تتخطى التجربة بسبب المكافأة فقط، قد تظل مفيدة

            # التحقق من الحالة والحالة التالية
            state_arr = np.asarray(state, dtype=np.float32)
            if np.isnan(state_arr).any() or np.isinf(state_arr).any():
                skip_experience = True
                self._push_nan_warnings['state'] += 1
                nan_source = 'state'

            next_state_arr = np.asarray(next_state, dtype=np.float32)
            if not skip_experience and (np.isnan(next_state_arr).any() or np.isinf(next_state_arr).any()):
                skip_experience = True
                self._push_nan_warnings['next_state'] += 1
                nan_source = 'next_state'

            # التحقق من الإجراء
            action_arr = np.asarray(action, dtype=np.float32)
            if not skip_experience and (np.isnan(action_arr).any() or np.isinf(action_arr).any()):
                skip_experience = True
                self._push_nan_warnings['action'] += 1
                nan_source = 'action'

            # التحقق من done (تحويل إلى float)
            done_float = float(done)

            if skip_experience:
                # طباعة تحذير واحد كل فترة لتجنب إغراق السجل
                total_warnings = sum(self._push_nan_warnings.values())
                if total_warnings % 500 == 1: # طباعة كل 500 تحذير
                    warnings.warn(f"Skipping experience due to NaN/Inf detected in '{nan_source}' (Total push skips: S:{self._push_nan_warnings['state']}, A:{self._push_nan_warnings['action']}, R(fixed):{self._push_nan_warnings['reward']}, S':{self._push_nan_warnings['next_state']}).", RuntimeWarning)
                return

            # تخزين كمصفوفات NumPy للحفاظ على الذاكرة قليلاً
            experience = (state_arr, action_arr, float(reward), next_state_arr, done_float)
            self.buffer.append(experience)

        except (TypeError, ValueError) as e:
            warnings.warn(f"Could not process or store experience: {e}. Skipping.", RuntimeWarning)

    def sample(self, batch_size):
        """
        يأخذ عينة عشوائية ويحولها إلى تنسورات PyTorch مع فحص NaN.
        """
        current_size = len(self.buffer)
        if current_size < batch_size:
            return None

        try:
            batch_indices = np.random.choice(current_size, batch_size, replace=False)
            batch = [self.buffer[i] for i in batch_indices]
            # بديل: batch = random.sample(self.buffer, batch_size) # قد يكون أبطأ للـ deque الكبيرة
        except ValueError:
             # قد يحدث إذا كان batch_size > current_size (تم التحقق منه بالفعل) أو مشكلة أخرى
             warnings.warn(f"ValueError during buffer sampling (requested {batch_size}, have {current_size}).", RuntimeWarning)
             return None
        except Exception as e:
             warnings.warn(f"Unexpected error during buffer sampling: {e}.", RuntimeWarning)
             return None

        # فصل المكونات وتحويلها
        try:
            states, actions, rewards, next_states, dones = zip(*batch)

            # تحويل إلى مصفوفات NumPy دفعة واحدة
            states_np = np.array(states, dtype=np.float32)
            actions_np = np.array(actions, dtype=np.float32)
            rewards_np = np.array(rewards, dtype=np.float32).reshape(-1, 1) # (BatchSize, 1)
            next_states_np = np.array(next_states, dtype=np.float32)
            dones_np = np.array(dones, dtype=np.float32).reshape(-1, 1) # (BatchSize, 1)

            # --- فحص NaN/Inf في مصفوفات NumPy قبل التحويل إلى تنسورات ---
            # هذا الفحص مهم لأن NaN في الدفعة يمكن أن يفسد التحديث بأكمله
            if np.isnan(states_np).any() or np.isnan(actions_np).any() or \
               np.isnan(rewards_np).any() or np.isnan(next_states_np).any() or \
               np.isnan(dones_np).any() or \
               np.isinf(states_np).any() or np.isinf(actions_np).any() or \
               np.isinf(rewards_np).any() or np.isinf(next_states_np).any() or \
               np.isinf(dones_np).any():

                self._sample_nan_warning += 1
                if self._sample_nan_warning % 100 == 1: # طباعة تحذير كل 100 مرة
                    warnings.warn(f"NaN/Inf detected in sampled NumPy batch before tensor conversion (Total sample skips: {self._sample_nan_warning}). Returning None.", RuntimeWarning)
                # تحديد مصدر الـ NaN/Inf للمساعدة في التشخيص
                # if np.isnan(states_np).any() or np.isinf(states_np).any(): print("  (NaN/Inf found in states_np)")
                # if np.isnan(actions_np).any() or np.isinf(actions_np).any(): print("  (NaN/Inf found in actions_np)")
                # ... etc ...
                return None

            # تحويل إلى تنسورات (على CPU افتراضيًا، سيتم نقلها إلى الجهاز في الوكيل)
            states_tensor = torch.from_numpy(states_np)
            actions_tensor = torch.from_numpy(actions_np)
            rewards_tensor = torch.from_numpy(rewards_np)
            next_states_tensor = torch.from_numpy(next_states_np)
            dones_tensor = torch.from_numpy(dones_np)

        except (ValueError, TypeError, Exception) as e:
            warnings.warn(f"Failed to convert sampled batch to tensors: {e}. Returning None.", RuntimeWarning)
            return None

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self):
        return len(self.buffer)

# --- 6. Core Component: RL-E² Agent (DDPG-like) ---

class RLE2Agent:
    """
    وكيل RL-E²: يستخدم معادلات متطورة للسياسة (Actor) والمقيّم (Critic - Q-function)،
    ويطبق تحديث Actor-Critic شبيه بـ DDPG مع شبكات هدف وتطور المعادلات.
    """
    def __init__(self, state_dim, action_dim, action_bounds,
                 policy_lr=5e-4, q_lr=5e-4, gamma=0.99, tau=0.005, # tau لتحديث الهدف
                 buffer_capacity=int(1e5), batch_size=128,
                 init_policy_complexity=4, init_q_complexity=5, # تعقيد أولي مختلف؟
                 policy_complexity_limit=15, q_complexity_limit=15,
                 exploration_noise_std=0.3, noise_decay_rate=0.9998, min_exploration_noise=0.1, # قيم مقترحة
                 policy_mutation=0.03, q_mutation=0.03, # قوى تحور
                 weight_decay=1e-5, grad_clip_norm=1.0):
        """
        تهيئة وكيل RL-E² ببنية DDPG-like.

        Args:
           state_dim (int): أبعاد فضاء الحالة.
           action_dim (int): أبعاد فضاء الإجراء.
           action_bounds (tuple): (min_action, max_action) لحدود الإجراء.
           policy_lr (float): معدل تعلم السياسة (Actor).
           q_lr (float): معدل تعلم المقيّم (Critic - Q-function).
           gamma (float): معامل الخصم.
           tau (float): معامل التحديث الناعم لشبكات الهدف (Polyak averaging).
           buffer_capacity (int): سعة ذاكرة التجارب.
           batch_size (int): حجم دفعة التدريب.
           init_policy_complexity (int): التعقيد الأولي لمعادلة السياسة.
           init_q_complexity (int): التعقيد الأولي لمعادلة المقيّم.
           policy_complexity_limit (int): الحد الأقصى لتعقيد السياسة.
           q_complexity_limit (int): الحد الأقصى لتعقيد المقيّم.
           exploration_noise_std (float): الانحراف المعياري لضوضاء الاستكشاف الأولية.
           noise_decay_rate (float): معامل تناقص ضوضاء الاستكشاف.
           min_exploration_noise (float): الحد الأدنى لضوضاء الاستكشاف.
           policy_mutation (float): قوة التحور الأساسية للسياسة.
           q_mutation (float): قوة التحور الأساسية للمقيّم.
           weight_decay (float): معامل اضمحلال الوزن (L2 regularization).
           grad_clip_norm (float): القيمة القصوى لـ norm التدرجات (للتقليم). <= 0 لتعطيل التقليم.
        """
        # --- 1. تحديد الجهاز وتخزين المعلمات ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"INFO: Agent using device: {self.device}")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.grad_clip_norm = grad_clip_norm if grad_clip_norm > 0 else None # None لتعطيل التقليم

        # معالجة حدود الإجراء
        if action_bounds is None or len(action_bounds) != 2:
            raise ValueError("action_bounds must be a tuple of (min_action, max_action).")
        # ضمان أن الحدود هي أرقام عشرية
        self.action_low = float(action_bounds[0])
        self.action_high = float(action_bounds[1])
        if self.action_low >= self.action_high:
            raise ValueError("action_low must be less than action_high.")

        # حساب مقياس وانحياز الإجراء (لنقل مخرج tanh [-1, 1] إلى [low, high])
        # التأكد من أن العمليات تتم باستخدام float32
        self.action_scale = torch.tensor((self.action_high - self.action_low) / 2.0, dtype=torch.float32, device=self.device)
        self.action_bias = torch.tensor((self.action_high + self.action_low) / 2.0, dtype=torch.float32, device=self.device)
        # التأكد من أن المقياس موجب تمامًا لتجنب القسمة على صفر أو مشاكل أخرى
        if (self.action_scale <= 1e-6).any(): # التحقق من كل عنصر إذا كان المقياس متعدد الأبعاد
             warnings.warn(f"Calculated action_scale ({self.action_scale}) has zero or near-zero elements. Check action bounds.", RuntimeWarning)
             # قد تحتاج إلى معالجة خاصة هنا، مثل استبدال الأصفار بقيمة صغيرة
             self.action_scale = torch.clamp(self.action_scale, min=1e-6)


        # --- 2. إنشاء المعادلات المتطورة وشبكات الهدف ---

        # Actor (Policy) Network
        self.policy_eq = EvolvingEquation(
            input_dim=state_dim, init_complexity=init_policy_complexity, output_dim=action_dim,
            complexity_limit=policy_complexity_limit, output_activation=None # Tanh يطبق خارج المعادلة
        ).to(self.device)
        self.target_policy_eq = copy.deepcopy(self.policy_eq) # إنشاء نسخة طبق الأصل للهدف
        # تجميد معلمات شبكة الهدف في البداية
        for param in self.target_policy_eq.parameters():
             param.requires_grad = False

        # Critic (Q-Function) Network
        q_input_dim = state_dim + action_dim
        self.q_eq = EvolvingEquation(
            input_dim=q_input_dim, init_complexity=init_q_complexity, output_dim=1,
            complexity_limit=q_complexity_limit, output_activation=None # Q-values لا تحتاج تنشيط مقيد عادة
        ).to(self.device)
        self.target_q_eq = copy.deepcopy(self.q_eq) # إنشاء نسخة طبق الأصل للهدف
        # تجميد معلمات شبكة الهدف في البداية
        for param in self.target_q_eq.parameters():
            param.requires_grad = False

        print("\n--- Initial Evolving Equations (DDPG-like) ---")
        print(f"Policy Eq (Actor): Complexity={self.policy_eq.complexity}, Limit={self.policy_eq.complexity_limit}, InputDim={self.policy_eq.input_dim}, OutputDim={self.policy_eq.output_dim}")
        print(f"Q Eq (Critic):     Complexity={self.q_eq.complexity}, Limit={self.q_eq.complexity_limit}, InputDim={self.q_eq.input_dim}, OutputDim={self.q_eq.output_dim}")
        print("-" * 50 + "\n")

        # --- 3. إنشاء محركات التطور ---
        # فترات تبريد أطول قليلاً قد تكون مفيدة بعد التغييرات الهيكلية
        self.policy_evolver = EvolutionEngine(mutation_power=policy_mutation, cooldown_period=40, history_size=60)
        self.q_evolver = EvolutionEngine(mutation_power=q_mutation, cooldown_period=40, history_size=60)

        # --- 4. إنشاء المحسنات ---
        # استخدام AdamW (Adam with decoupled weight decay)
        try:
            self.policy_optimizer = optim.AdamW(self.policy_eq.parameters(), lr=policy_lr, weight_decay=weight_decay)
            self.q_optimizer = optim.AdamW(self.q_eq.parameters(), lr=q_lr, weight_decay=weight_decay)
        except ValueError as e: # قد يحدث إذا كانت المعادلة فارغة (نادر جدًا)
             raise RuntimeError(f"Failed to initialize optimizers, possibly due to empty initial equations: {e}")


        # --- 5. إنشاء ذاكرة التجارب ---
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # --- 6. إعداد معلمات الاستكشاف ---
        self.exploration_noise_std = exploration_noise_std
        self.min_exploration_noise = min_exploration_noise
        self.noise_decay_rate = noise_decay_rate

        # --- 7. إحصائيات التتبع ---
        self.policy_struct_changes = 0
        self.q_struct_changes = 0
        self.total_updates = 0

    def _update_target_network(self, main_net, target_net):
        """ينفذ تحديثًا ناعمًا (Polyak averaging) لشبكة الهدف."""
        with torch.no_grad():
            for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * main_param.data)

    def _sync_target_network(self, main_net, target_net):
        """ ينسخ أوزان الشبكة الرئيسية إلى شبكة الهدف بالكامل.
            يستخدم عند تغيير هيكل الشبكة الرئيسية. """
        with torch.no_grad():
            # الطريقة الأسهل هي إنشاء نسخة عميقة جديدة
             target_net = copy.deepcopy(main_net)
             # تجميد معلمات الهدف الجديدة
             for param in target_net.parameters():
                 param.requires_grad = False
        return target_net


    def get_action(self, state, explore=True):
        """
        يختار إجراءً بناءً على الحالة الحالية، مع إضافة ضوضاء استكشاف (إذا مفعلة)،
        وتطبيق tanh، والتكييف، والتقييد. يتضمن فحص NaN.
        """
        # --- 1. تحويل الحالة وفحص NaN ---
        try:
            # تأكد من أن الحالة هي مصفوفة NumPy float32
            state_np = np.asarray(state, dtype=np.float32)
            if np.isnan(state_np).any() or np.isinf(state_np).any():
                warnings.warn(f"NaN/Inf detected in state input to get_action: {state_np}. Using zeros.", RuntimeWarning)
                state_np = np.zeros_like(state_np, dtype=np.float32)
            state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
        except Exception as e:
            warnings.warn(f"Failed to process state in get_action: {e}. Returning zero action.", RuntimeWarning)
            # إعادة إجراء صفري بالشكل الصحيح وفي النطاق الصحيح
            zero_action = np.zeros(self.action_dim, dtype=np.float32)
            # استخدام الانحياز لإرجاع قيمة في منتصف النطاق قد يكون أفضل
            # mid_action = self.action_bias.cpu().numpy()
            # return mid_action.reshape(self.action_dim)
            return zero_action # أو إرجاع الصفر إذا كان مناسبًا

        # --- 2. الحصول على الإجراء الأولي من السياسة (بدون تدرجات) ---
        self.policy_eq.eval() # وضع التقييم مهم لإيقاف Dropout/BatchNorm إذا استُخدما
        action_raw = None
        try:
            with torch.no_grad():
                action_raw = self.policy_eq(state_tensor) # المخرج الأولي (قبل tanh) - Shape: (1, ActionDim)
                if torch.isnan(action_raw).any() or torch.isinf(action_raw).any():
                    warnings.warn(f"NaN/Inf detected in raw policy output (Actor): {action_raw}. Clamping to 0.", RuntimeWarning)
                    action_raw = torch.zeros_like(action_raw)
        except Exception as e:
             warnings.warn(f"Exception during policy_eq forward in get_action: {e}. Returning zero action.", RuntimeWarning)
             self.policy_eq.train() # إعادة للتدريب
             zero_action = np.zeros(self.action_dim, dtype=np.float32)
             return zero_action

        # --- 3. تطبيق Tanh للحصول على إجراء في [-1, 1] ---
        action_tanh = torch.tanh(action_raw) # Shape: (1, ActionDim)

        # --- 4. إضافة ضوضاء الاستكشاف (إذا كان مفعلًا) *بعد* tanh ---
        #    (في DDPG الأصلي، الضوضاء تضاف *قبل* التقييد النهائي، يمكن تجربته بعد tanh أو على action_raw)
        #    لنضفها هنا على action_tanh [-1, 1] ثم نقيدها.
        if explore:
            # إنشاء ضوضاء بنفس الشكل والجهاز
            noise = torch.randn(self.action_dim, device=self.device) * self.exploration_noise_std
            action_noisy_tanh = action_tanh + noise
            # تقييد الضوضاء ضمن نطاق [-1, 1] للحفاظ على تأثير tanh
            action_noisy_tanh = torch.clamp(action_noisy_tanh, -1.0, 1.0)
        else:
            action_noisy_tanh = action_tanh

        # --- 5. تكييف النطاق والتقييد النهائي ---
        try:
            # تكييف النطاق باستخدام المقياس والانحياز
            action_scaled = action_noisy_tanh * self.action_scale + self.action_bias

            # تقييد الإجراء النهائي ضمن حدود البيئة (ضروري جدًا)
            action_clipped = torch.clamp(action_scaled, self.action_low, self.action_high)

            # فحص NaN/Inf النهائي بعد كل العمليات
            if torch.isnan(action_clipped).any() or torch.isinf(action_clipped).any():
                 warnings.warn(f"NaN/Inf detected in final clipped action: {action_clipped}. Returning middle action.", RuntimeWarning)
                 # قيمة متوسطة آمنة
                 action_clipped = torch.full_like(action_clipped, (self.action_high + self.action_low) / 2.0)

            # تحويل إلى numpy على CPU
            final_action = action_clipped.squeeze(0).cpu().numpy()

        except Exception as e:
            warnings.warn(f"Exception during action scaling/clipping: {e}. Returning zero action.", RuntimeWarning)
            final_action = np.zeros(self.action_dim, dtype=np.float32)

        # --- 6. إعادة المعادلة إلى وضع التدريب ---
        self.policy_eq.train()

        # التأكد من أن الشكل صحيح (خاصة للبيئات ذات بعد إجراء واحد)
        return final_action.reshape(self.action_dim)

    def update(self, step):
        """
        ينفذ خطوة تحديث واحدة للوكيل (Actor-Critic DDPG-like) وتطور المعادلات.
        يتضمن فحص NaN وتقليم التدرج وتحديث شبكات الهدف.
        """
        # --- 1. التحقق وأخذ العينة من الذاكرة ---
        if len(self.replay_buffer) < self.batch_size:
            return None, None, 0.0 # لا خسائر، لا متوسط مكافأة

        self.total_updates += 1
        sample = self.replay_buffer.sample(self.batch_size)
        if sample is None:
            # فشل أخذ العينة (غالبًا بسبب NaN كما تم اكتشافه في sample())
            # warnings.warn(f"Failed to sample from buffer at update step {step}. Skipping update.", RuntimeWarning)
            return None, None, 0.0

        # نقل التنسورات إلى الجهاز المحدد
        try:
            states, actions, rewards, next_states, dones = [t.to(self.device) for t in sample]
        except Exception as e:
            warnings.warn(f"Failed to move sampled batch to device {self.device}: {e}. Skipping update.", RuntimeWarning)
            return None, None, 0.0

        # حساب متوسط المكافأة في الدفعة (للتطور)
        avg_reward_in_batch = rewards.mean().item()
        if math.isnan(avg_reward_in_batch) or math.isinf(avg_reward_in_batch):
             # warnings.warn(f"NaN/Inf average reward in batch at update step {step}. Using 0 for evolution.", RuntimeWarning)
             avg_reward_in_batch = 0.0 # قيمة آمنة

        q_loss_item = None
        policy_loss_item = None

        # --- 2. تحديث المقيّم (Critic / Q-function Update) ---
        try:
            self.q_eq.train() # وضع التدريب للمقيّم

            # --- أ. حساب الإجراءات التالية من شبكة السياسة الهدف ---
            with torch.no_grad(): # لا تدرجات للهدف
                next_actions_raw = self.target_policy_eq(next_states)
                # تطبيق tanh والتكييف للحصول على الإجراءات في النطاق الصحيح
                next_actions_tanh = torch.tanh(next_actions_raw)
                next_actions_scaled = next_actions_tanh * self.action_scale + self.action_bias
                # تقييد إجراءات الهدف ضمن الحدود (مهم!)
                next_actions_clipped = torch.clamp(next_actions_scaled, self.action_low, self.action_high)

                # فحص NaN/Inf في إجراءات الهدف
                if torch.isnan(next_actions_clipped).any() or torch.isinf(next_actions_clipped).any():
                     warnings.warn(f"NaN/Inf detected in target policy actions at step {step}. Clamping to zero.", RuntimeWarning)
                     next_actions_clipped = torch.zeros_like(next_actions_clipped)


                # --- ب. حساب قيم Q للهدف باستخدام شبكة المقيّم الهدف ---
                # دمج الحالة التالية والإجراء التالي كمدخل للمقيّم الهدف
                target_q_inputs = torch.cat([next_states, next_actions_clipped], dim=1)
                target_q_values = self.target_q_eq(target_q_inputs)

                if torch.isnan(target_q_values).any() or torch.isinf(target_q_values).any():
                     warnings.warn(f"NaN/Inf in target_q_values Q(s', a') at step {step}. Clamping target to 0.", RuntimeWarning)
                     target_q_values = torch.nan_to_num(target_q_values, nan=0.0, posinf=0.0, neginf=0.0)

                # --- ج. حساب هدف Q (معادلة بلمان) ---
                q_target = rewards + self.gamma * target_q_values * (1.0 - dones)

                if torch.isnan(q_target).any() or torch.isinf(q_target).any():
                     warnings.warn(f"NaN/Inf in final q_target at step {step}. Skipping Q update.", RuntimeWarning)
                     raise ValueError("NaN/Inf in q_target") # للانتقال إلى كتلة except

            # --- د. حساب قيم Q الحالية باستخدام شبكة المقيّم الرئيسية ---
            # دمج الحالة الحالية والإجراء الفعلي من الدفعة كمدخل للمقيّم الرئيسي
            current_q_inputs = torch.cat([states, actions], dim=1)
            current_q_values = self.q_eq(current_q_inputs)

            if torch.isnan(current_q_values).any() or torch.isinf(current_q_values).any():
                 warnings.warn(f"NaN/Inf in current_q_values Q(s, a) at step {step}. Skipping Q update.", RuntimeWarning)
                 raise ValueError("NaN/Inf in current_q_values") # للانتقال إلى كتلة except

            # --- هـ. حساب خسارة المقيّم (MSE) ---
            # استخدم q_target.detach() للتأكد من عدم تدفق التدرجات من الهدف
            q_loss = F.mse_loss(current_q_values, q_target.detach())

            # --- و. تحديث المقيّم ---
            if torch.isnan(q_loss) or torch.isinf(q_loss):
                warnings.warn(f"NaN/Inf Q-loss detected at step {step}: {q_loss.item()}. Skipping Q optimizer step.", RuntimeWarning)
                # print(f"  CurrentQ mean: {current_q_values.mean().item():.2f}, TargetQ mean: {q_target.mean().item():.2f}")
            else:
                self.q_optimizer.zero_grad()
                q_loss.backward()
                if self.grad_clip_norm:
                    nn.utils.clip_grad_norm_(self.q_eq.parameters(), max_norm=self.grad_clip_norm)
                self.q_optimizer.step()
                q_loss_item = q_loss.item()

        except Exception as e:
            # إذا فشل تحديث المقيّم، لا تقم بتحديث السياسة لمنع تفاقم المشكلة
            warnings.warn(f"Exception during Critic (Q) update at step {step}: {e}", RuntimeWarning)
            # لا تقم بتحديث الهدف أو التطور إذا فشل المقيّم
            return q_loss_item, policy_loss_item, avg_reward_in_batch

        # --- 3. تحديث السياسة (Actor Update) ---
        # يتم تحديث السياسة بشكل أقل تكرارًا في بعض تطبيقات DDPG (e.g., TD3)
        # لكن هنا سنحدثها في كل خطوة مثل المقيّم للتبسيط
        try:
            self.policy_eq.train() # وضع التدريب للسياسة

            # --- أ. تجميد المقيّم (لا نريد تحديثه بخسارة السياسة) ---
            # هذا ليس ضروريًا تمامًا إذا استخدمنا .detach() بشكل صحيح،
            # ولكنه يوضح النية وقد يمنع أخطاء غير متوقعة.
            for param in self.q_eq.parameters():
                param.requires_grad = False

            # --- ب. حساب الإجراءات المقترحة من السياسة الرئيسية ---
            actions_pred_raw = self.policy_eq(states)
            # تطبيق tanh والتكييف (نفس ما يحدث في get_action بدون ضوضاء)
            actions_pred_tanh = torch.tanh(actions_pred_raw)
            actions_pred_scaled = actions_pred_tanh * self.action_scale + self.action_bias
            # لا حاجة للتقييد هنا عادةً، لأننا نريد أن تتعلم السياسة المخرجات الصحيحة

            # --- ج. حساب خسارة السياسة (تعظيم -Q) ---
            # دمج الحالة والإجراءات المقترحة لإدخالها للمقيّم
            policy_q_inputs = torch.cat([states, actions_pred_scaled], dim=1)
            # استخدام المقيّم الرئيسي (q_eq) لتقييم الإجراءات المقترحة
            policy_q_values = self.q_eq(policy_q_inputs)

            # الخسارة هي سالب متوسط قيم Q (لأننا نريد تعظيم Q)
            policy_loss = -policy_q_values.mean()

            # --- د. تحديث السياسة ---
            if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                 warnings.warn(f"NaN/Inf Policy loss detected at step {step}: {policy_loss.item()}. Skipping Policy optimizer step.", RuntimeWarning)
            else:
                self.policy_optimizer.zero_grad()
                policy_loss.backward() # التدرجات تتدفق الآن من Q(s, policy(s)) إلى policy_eq
                if self.grad_clip_norm:
                    nn.utils.clip_grad_norm_(self.policy_eq.parameters(), max_norm=self.grad_clip_norm)
                self.policy_optimizer.step()
                policy_loss_item = policy_loss.item()

            # --- هـ. إعادة تفعيل تدرجات المقيّم ---
            for param in self.q_eq.parameters():
                param.requires_grad = True

        except Exception as e:
            warnings.warn(f"Exception during Actor (Policy) update at step {step}: {e}", RuntimeWarning)
            # التأكد من إعادة تفعيل تدرجات المقيّم حتى لو فشل تحديث السياسة
            try:
                for param in self.q_eq.parameters():
                    param.requires_grad = True
            except Exception as e_reraise:
                 warnings.warn(f"Failed to re-enable Q-gradients after policy update failure: {e_reraise}", RuntimeWarning)
            # لا تقم بتحديث الهدف أو التطور إذا فشلت السياسة
            return q_loss_item, policy_loss_item, avg_reward_in_batch

        # --- 4. تحديث شبكات الهدف (Polyak Averaging) ---
        try:
            self._update_target_network(self.policy_eq, self.target_policy_eq)
            self._update_target_network(self.q_eq, self.target_q_eq)
        except Exception as e:
             warnings.warn(f"Exception during target network update at step {step}: {e}", RuntimeWarning)


        # --- 5. تطور المعادلات (بعد التحديثات الرئيسية وتحديث الهدف) ---
        structure_changed_policy = False
        structure_changed_q = False
        try:
            # تطور السياسة (مرر المحسن الخاص بها)
            self.policy_optimizer, structure_changed_policy = self.policy_evolver.evolve(
                self.policy_eq, avg_reward_in_batch, self.total_updates, self.policy_optimizer
            )
            if structure_changed_policy:
                self.policy_struct_changes += 1
                # *** مهم: مزامنة شبكة الهدف بعد تغيير هيكل الشبكة الرئيسية ***
                print(f"    Syncing target policy network due to structure change.")
                self.target_policy_eq = self._sync_target_network(self.policy_eq, self.target_policy_eq)


            # تطور المقيّم (مرر المحسن الخاص به)
            self.q_optimizer, structure_changed_q = self.q_evolver.evolve(
                self.q_eq, avg_reward_in_batch, self.total_updates, self.q_optimizer # استخدم avg_reward كبديل لخسارة Q كمؤشر أداء
            )
            if structure_changed_q:
                self.q_struct_changes += 1
                # *** مهم: مزامنة شبكة الهدف بعد تغيير هيكل الشبكة الرئيسية ***
                print(f"    Syncing target Q network due to structure change.")
                self.target_q_eq = self._sync_target_network(self.q_eq, self.target_q_eq)


        except Exception as e:
            warnings.warn(f"Exception during evolution step {self.total_updates}: {e}", RuntimeWarning)
            import traceback
            traceback.print_exc() # طباعة التتبع للمساعدة في التشخيص


        # --- 6. تقليل ضوضاء الاستكشاف ---
        self.exploration_noise_std = max(self.min_exploration_noise,
                                         self.exploration_noise_std * self.noise_decay_rate)

        return q_loss_item, policy_loss_item, avg_reward_in_batch

    def evaluate(self, env, episodes=5):
        """
        يقيم أداء الوكيل الحالي (بدون استكشاف).
        يتضمن معالجة NaN وفحص أخطاء البيئة.
        """
        total_rewards = []
        # محاولة الحصول على أقصى خطوات من البيئة، مع قيمة افتراضية آمنة
        try:
            max_episode_steps = env.spec.max_episode_steps if env.spec and env.spec.max_episode_steps else 1000
        except AttributeError:
            max_episode_steps = 1000 # قيمة افتراضية إذا لم يتم العثور على spec

        eval_device = self.device # استخدام نفس جهاز الوكيل للتقييم

        for i in range(episodes):
            episode_reward = 0.0
            steps_in_episode = 0
            state, info = None, None # تهيئة
            try:
                # استخدام بذرة مختلفة لكل حلقة تقييم لزيادة التنوع
                eval_seed = RANDOM_SEED + 1000 + i + self.total_updates
                state, info = env.reset(seed=eval_seed)
                # التأكد من أن الحالة الأولية بالتنسيق الصحيح
                state = np.asarray(state, dtype=np.float32)
            except Exception as e:
                 warnings.warn(f"Failed to reset evaluation environment for episode {i+1}: {e}. Skipping episode.", RuntimeWarning)
                 continue

            terminated = False
            truncated = False

            while not (terminated or truncated):
                # التحقق من الحد الأقصى للخطوات يدويًا
                if steps_in_episode >= max_episode_steps:
                    # print(f"DEBUG: Eval episode {i+1} truncated at {max_episode_steps} steps.")
                    truncated = True # إنهاء الحلقة بسبب الحد الأقصى للخطوات
                    break

                try:
                    # اختيار الإجراء بدون استكشاف
                    action = self.get_action(state, explore=False)

                    # فحص الإجراء قبل تمريره للبيئة
                    if action is None or np.isnan(action).any() or np.isinf(action).any():
                         warnings.warn(f"Invalid action generated during evaluation ep {i+1}: {action}. Using zero action.", RuntimeWarning)
                         action = np.zeros(self.action_dim, dtype=np.float32)


                    # التفاعل مع البيئة
                    next_state, reward, terminated, truncated, info = env.step(action)

                    # التحقق من المكافأة
                    if math.isnan(reward) or math.isinf(reward):
                        # warnings.warn(f"NaN/Inf reward ({reward}) during evaluation ep {i+1}. Using 0.", RuntimeWarning)
                        reward = 0.0

                    episode_reward += reward
                    # التأكد من أن الحالة التالية بالتنسيق الصحيح
                    state = np.asarray(next_state, dtype=np.float32)
                    steps_in_episode += 1

                except (gym.error.Error, Exception) as e: # التقاط أخطاء Gym المحتملة أيضًا
                    warnings.warn(f"Exception during evaluation step in ep {i+1}: {e}. Ending episode.", RuntimeWarning)
                    terminated = True # إنهاء الحلقة عند الخطأ

            total_rewards.append(episode_reward)

        if not total_rewards:
             warnings.warn(f"Evaluation failed for all {episodes} episodes.", RuntimeWarning)
             return -np.inf # إرجاع قيمة سيئة جدًا

        mean_reward = np.mean(total_rewards)
        return mean_reward

    def save_model(self, filename="rle2_agent_checkpoint.pt"):
        """يحفظ حالة الوكيل، بما في ذلك المعادلات الأربعة، المحسنات، التعقيد، وحالة التدريب."""
        # نقل الشبكات إلى CPU للحفظ (أفضل توافقية)
        self.policy_eq.to('cpu')
        self.target_policy_eq.to('cpu')
        self.q_eq.to('cpu')
        self.target_q_eq.to('cpu')

        try:
            # تمثيل الدوال (كمرجع)
            policy_funcs_repr = [f.__name__ if hasattr(f, '__name__') else repr(f) for f in self.policy_eq.functions]
            q_funcs_repr = [f.__name__ if hasattr(f, '__name__') else repr(f) for f in self.q_eq.functions]

            save_data = {
                'metadata': {
                    'description': 'RL-E² Agent Checkpoint (v1.2 - DDPG-like)',
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'random_seed': RANDOM_SEED,
                    'torch_version': torch.__version__,
                    'gym_version': gym.__version__,
                },
                'agent_config': { # حفظ إعدادات الوكيل الأساسية
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim,
                    'action_bounds': (self.action_low, self.action_high),
                    'gamma': self.gamma,
                    'tau': self.tau,
                    'batch_size': self.batch_size,
                    'policy_complexity_limit': self.policy_eq.complexity_limit,
                    'q_complexity_limit': self.q_eq.complexity_limit,
                    # حفظ معدلات التعلم الأصلية قد يكون مفيدًا
                    'policy_lr': self.policy_optimizer.param_groups[0]['lr'],
                    'q_lr': self.q_optimizer.param_groups[0]['lr'],
                    'weight_decay': self.policy_optimizer.param_groups[0].get('weight_decay', 0), # افتراض أنها متساوية
                },
                # حفظ حالة المعادلات الرئيسية فقط (الهدف يعاد بناؤه)
                'policy_eq_state_dict': self.policy_eq.state_dict(),
                'q_eq_state_dict': self.q_eq.state_dict(),
                # حفظ حالة المحسنات
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                'q_optimizer_state_dict': self.q_optimizer.state_dict(),
                # حفظ التعقيد الحالي للمعادلات
                'policy_complexity': self.policy_eq.complexity,
                'q_complexity': self.q_eq.complexity,
                # حفظ الدوال (كمرجع)
                'policy_functions_repr': policy_funcs_repr,
                'q_functions_repr': q_funcs_repr,
                'training_state': {
                    'exploration_noise_std': self.exploration_noise_std,
                    'min_exploration_noise': self.min_exploration_noise,
                    'noise_decay_rate': self.noise_decay_rate,
                    'total_updates': self.total_updates,
                    'policy_struct_changes': self.policy_struct_changes,
                    'q_struct_changes': self.q_struct_changes,
                },
                 # حفظ حالة محركات التطور (سجل الأداء)
                 'policy_evolver_state': {
                     'performance_history': list(self.policy_evolver.performance_history),
                     'term_change_cooldown': self.policy_evolver.term_change_cooldown, # حفظ فترة التبريد
                 },
                 'q_evolver_state': {
                      'performance_history': list(self.q_evolver.performance_history),
                      'term_change_cooldown': self.q_evolver.term_change_cooldown, # حفظ فترة التبريد
                 }
            }
            # إنشاء المجلد إذا لم يكن موجودًا
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            torch.save(save_data, filename)
            print(f"INFO: Agent state saved successfully to '{filename}'")

        except Exception as e:
            warnings.warn(f"Failed to save agent state to '{filename}': {e}", RuntimeWarning)
            import traceback
            traceback.print_exc()
        finally:
            # إعادة الشبكات إلى الجهاز الأصلي
            self.policy_eq.to(self.device)
            self.target_policy_eq.to(self.device)
            self.q_eq.to(self.device)
            self.target_q_eq.to(self.device)

    def load_model(self, filename="rle2_agent_checkpoint.pt"):
        """
        يحمل حالة الوكيل، ويعيد بناء المعادلات والمحسنات والشبكات المستهدفة
        إذا اختلف التعقيد المحفوظ أو عند التحميل الأولي.
        """
        try:
            print(f"INFO: Attempting to load agent state from '{filename}'...")
            checkpoint = torch.load(filename, map_location=self.device)

            # --- 1. التحقق من البيانات الأساسية ---
            required_keys = ['agent_config', 'training_state', 'policy_eq_state_dict',
                             'q_eq_state_dict', 'policy_complexity', 'q_complexity',
                             'policy_optimizer_state_dict', 'q_optimizer_state_dict']
            if not all(key in checkpoint for key in required_keys):
                missing_keys = [key for key in required_keys if key not in checkpoint]
                warnings.warn(f"Checkpoint file '{filename}' is incomplete. Missing keys: {missing_keys}. Load failed.", RuntimeWarning)
                return False

            cfg = checkpoint['agent_config']
            train_state = checkpoint['training_state']

            # --- 2. التحقق من التوافق الأساسي (الأبعاد، الحدود) ---
            if cfg.get('state_dim') != self.state_dim or cfg.get('action_dim') != self.action_dim:
                warnings.warn(f"Dimension mismatch! Agent (S:{self.state_dim}, A:{self.action_dim}) "
                      f"vs Checkpoint (S:{cfg.get('state_dim')}, A:{cfg.get('action_dim')}). Load aborted.", RuntimeWarning)
                return False
            saved_bounds = cfg.get('action_bounds')
            if saved_bounds and (abs(saved_bounds[0] - self.action_low) > 1e-6 or abs(saved_bounds[1] - self.action_high) > 1e-6):
                 warnings.warn(f"Action bounds mismatch. Agent: {(self.action_low, self.action_high)}, Checkpoint: {saved_bounds}. Continuing load, but check consistency.", RuntimeWarning)
                 # قد تحتاج إلى إعادة حساب self.action_scale و self.action_bias إذا كانت الحدود المحفوظة هي الصحيحة
                 # self.action_low, self.action_high = saved_bounds[0], saved_bounds[1]
                 # self.action_scale = torch.tensor(...)
                 # self.action_bias = torch.tensor(...)


            # --- 3. استعادة الإعدادات ومعدلات التعلم ---
            saved_policy_lr = cfg.get('policy_lr', self.policy_optimizer.param_groups[0]['lr'])
            saved_q_lr = cfg.get('q_lr', self.q_optimizer.param_groups[0]['lr'])
            saved_wd = cfg.get('weight_decay', self.policy_optimizer.param_groups[0].get('weight_decay', 0))
            self.gamma = cfg.get('gamma', self.gamma)
            self.tau = cfg.get('tau', self.tau)
            self.batch_size = cfg.get('batch_size', self.batch_size)


            # --- 4. إعادة بناء المعادلات والمحسنات إذا لزم الأمر ---
            #    (يتم هذا دائمًا عند التحميل لضمان أن البنية تطابق الحالة المحفوظة)
            saved_policy_comp = checkpoint.get('policy_complexity')
            saved_q_comp = checkpoint.get('q_complexity')

            print(f"  Loading model with Policy Complexity: {saved_policy_comp}, Q Complexity: {saved_q_comp}")

            try:
                # إعادة بناء السياسة (Actor)
                self.policy_eq = EvolvingEquation(
                    self.state_dim, init_complexity=saved_policy_comp, output_dim=self.action_dim,
                    complexity_limit=cfg.get('policy_complexity_limit', self.policy_eq.complexity_limit),
                    output_activation=None # Actor لا يستخدم تنشيط داخلي
                ).to(self.device)
                self.policy_eq.load_state_dict(checkpoint['policy_eq_state_dict'])
                self.policy_optimizer = optim.AdamW(self.policy_eq.parameters(), lr=saved_policy_lr, weight_decay=saved_wd)
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
                # إعادة بناء شبكة الهدف للسياسة ومزامنتها
                self.target_policy_eq = copy.deepcopy(self.policy_eq)
                for param in self.target_policy_eq.parameters(): param.requires_grad = False

                # إعادة بناء المقيّم (Critic)
                q_input_dim = self.state_dim + self.action_dim
                self.q_eq = EvolvingEquation(
                    q_input_dim, init_complexity=saved_q_comp, output_dim=1,
                    complexity_limit=cfg.get('q_complexity_limit', self.q_eq.complexity_limit),
                    output_activation=None # Critic لا يستخدم تنشيط
                ).to(self.device)
                self.q_eq.load_state_dict(checkpoint['q_eq_state_dict'])
                self.q_optimizer = optim.AdamW(self.q_eq.parameters(), lr=saved_q_lr, weight_decay=saved_wd)
                self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
                # إعادة بناء شبكة الهدف للمقيّم ومزامنتها
                self.target_q_eq = copy.deepcopy(self.q_eq)
                for param in self.target_q_eq.parameters(): param.requires_grad = False

            except (RuntimeError, KeyError, Exception) as e:
                 warnings.warn(f"ERROR rebuilding or loading state dicts: {e}. Load might be corrupted.", RuntimeWarning)
                 import traceback
                 traceback.print_exc()
                 return False


            # --- 5. استعادة حالة التدريب والتطور ---
            self.exploration_noise_std = train_state.get('exploration_noise_std', self.exploration_noise_std)
            self.min_exploration_noise = train_state.get('min_exploration_noise', self.min_exploration_noise)
            self.noise_decay_rate = train_state.get('noise_decay_rate', self.noise_decay_rate)
            self.total_updates = train_state.get('total_updates', self.total_updates)
            self.policy_struct_changes = train_state.get('policy_struct_changes', self.policy_struct_changes)
            self.q_struct_changes = train_state.get('q_struct_changes', self.q_struct_changes) # تحديث الاسم

            # استعادة حالة محركات التطور
            if 'policy_evolver_state' in checkpoint:
                 policy_evolver_state = checkpoint['policy_evolver_state']
                 policy_hist = policy_evolver_state.get('performance_history', [])
                 self.policy_evolver.performance_history = deque(policy_hist, maxlen=self.policy_evolver.performance_history.maxlen)
                 self.policy_evolver.term_change_cooldown = policy_evolver_state.get('term_change_cooldown', 0)
            if 'q_evolver_state' in checkpoint:
                 q_evolver_state = checkpoint['q_evolver_state']
                 q_hist = q_evolver_state.get('performance_history', [])
                 self.q_evolver.performance_history = deque(q_hist, maxlen=self.q_evolver.performance_history.maxlen)
                 self.q_evolver.term_change_cooldown = q_evolver_state.get('term_change_cooldown', 0)


            # استعادة تمثيل الدوال (كمرجع، لا يؤثر على التنفيذ الفعلي لكن مفيد للمعلومات)
            # policy_funcs_repr = checkpoint.get('policy_functions_repr')
            # q_funcs_repr = checkpoint.get('q_functions_repr')
            # print(f"  Loaded policy function representations (for info): {policy_funcs_repr}")
            # print(f"  Loaded Q function representations (for info): {q_funcs_repr}")

            print(f"INFO: Agent state loaded successfully from '{filename}'.")
            print(f"  Resumed Policy Complexity: {self.policy_eq.complexity} (Changes: {self.policy_struct_changes})")
            print(f"  Resumed Q Complexity: {self.q_eq.complexity} (Changes: {self.q_struct_changes})")
            print(f"  Resumed Exploration Noise: {self.exploration_noise_std:.4f}")
            print(f"  Resumed Total Updates: {self.total_updates}")
            return True

        except FileNotFoundError:
            warnings.warn(f"Checkpoint file not found at '{filename}'. Load failed.", RuntimeWarning)
            return False
        except Exception as e:
            warnings.warn(f"Unexpected error loading agent state from '{filename}': {e}", RuntimeWarning)
            import traceback
            traceback.print_exc()
            return False


# --- 7. Main Training Function ---

def train_rle2(env_name="Pendulum-v1", max_steps=100000, batch_size=128,
               eval_frequency=5000, start_learning_steps=2500, update_frequency=1,
               policy_lr=5e-4, q_lr=5e-4, tau=0.005, exploration_noise=0.3, noise_decay=0.9998, min_noise=0.1,
               save_best=True, save_best_metric='reward',
               save_periodically=False, save_interval=25000,
               render_eval=False, eval_episodes=5,
               resume_from_checkpoint=None, results_dir="rle2_results"):
    """
    الدالة الرئيسية لتدريب وكيل RL-E² (ببنية DDPG-like)، مع تحسينات في التتبع والحفظ والرسم البياني.
    """
    start_time = time.time()
    print("\n" + "="*60)
    print(f"=== Starting RL-E² Training (v1.2 DDPG-like) for {env_name} ===")
    print(f"=== Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print("="*60)

    print(f"\n--- Hyperparameters ---")
    print(f"  Environment: {env_name}, Max Steps: {max_steps:,}")
    print(f"  Batch Size: {batch_size}, Start Learning: {start_learning_steps:,}, Update Freq: {update_frequency}")
    print(f"  Eval Freq: {eval_frequency:,}, Eval Episodes: {eval_episodes}")
    print(f"  Policy LR: {policy_lr:.1e}, Q LR: {q_lr:.1e}, Tau: {tau:.3f}")
    print(f"  Exploration Noise Start: {exploration_noise:.2f}, Decay: {noise_decay:.4f}, Min Noise: {min_noise:.2f}")
    print(f"  Save Best: {save_best} (Metric: {save_best_metric}), Save Periodically: {save_periodically} (Interval: {save_interval:,})")
    print(f"  Resume from: {resume_from_checkpoint if resume_from_checkpoint else 'None'}")
    print(f"  Results Directory: {results_dir}")
    print("-" * 50 + "\n")


    # --- 1. تهيئة البيئة والوكيل ---
    try:
        # استخدام RecordEpisodeStatistics لتتبع المكافآت والأطوال تلقائيًا
        env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_name), deque_size=50) # تتبع آخر 50 حلقة
        eval_render_mode = "human" if render_eval else None
        # تهيئة بيئة التقييم
        try:
             eval_env = gym.make(env_name, render_mode=eval_render_mode)
        except TypeError: # للبيئات القديمة التي لا تقبل render_mode في __init__
             eval_env = gym.make(env_name)
             if render_eval: print("Warning: render_mode in gym.make() failed for eval_env, rendering might not work as expected.")
        except Exception as e:
             print(f"Warning: Could not create eval_env with render_mode='{eval_render_mode}'. Trying without. Error: {e}")
             eval_env = gym.make(env_name)


        # تعيين البذور للبيئات
        env.reset(seed=RANDOM_SEED)
        env.action_space.seed(RANDOM_SEED)
        eval_env.reset(seed=RANDOM_SEED + 1)
        eval_env.action_space.seed(RANDOM_SEED + 1)

        # الحصول على أبعاد الحالة والإجراء والحدود
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low
        action_high = env.action_space.high

        # التعامل مع الحدود غير المقيدة (مثل بعض بيئات التحكم)
        if np.any(np.isinf(action_low)) or np.any(np.isinf(action_high)):
             warnings.warn(f"Unbounded action space detected in '{env_name}'. Using default bounds [-1, 1]. This might limit performance if the actual range is larger.", RuntimeWarning)
             action_bounds = (-1.0, 1.0)
        else:
            # استخدام الحد الأدنى والأقصى عبر جميع أبعاد الإجراء
            action_bounds = (float(action_low.min()), float(action_high.max()))

        print(f"Environment Details: State Dim={state_dim}, Action Dim={action_dim}, Action Bounds={action_bounds}")

        # إنشاء الوكيل بالإعدادات المحدثة
        agent = RLE2Agent(
            state_dim, action_dim, action_bounds,
            policy_lr=policy_lr, q_lr=q_lr, tau=tau,
            batch_size=batch_size,
            exploration_noise_std=exploration_noise,
            noise_decay_rate=noise_decay,
            min_exploration_noise=min_noise
            # يمكن تعديل معلمات تعقيد الوكيل أو قوة التحور هنا إذا لزم الأمر
            # init_policy_complexity=..., init_q_complexity=..., etc.
        )

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize environment or agent: {e}")
        import traceback
        traceback.print_exc()
        # إغلاق البيئات إذا تم إنشاؤها جزئيًا
        if 'env' in locals() and env is not None: env.close()
        if 'eval_env' in locals() and eval_env is not None: eval_env.close()
        return None, []

    # --- 2. تهيئة متغيرات التتبع والتحكم ---
    evaluation_rewards = []
    steps_history = []
    # استخدام deque لتخزين الخسائر بكفاءة
    # حجم مناسب لتخزين خسائر فترة تقييم واحدة أو اثنتين
    loss_buffer_size = max(500, eval_frequency * 2 // update_frequency)
    all_q_losses = deque(maxlen=loss_buffer_size)
    all_policy_losses = deque(maxlen=loss_buffer_size)

    best_eval_metric_value = -np.inf # المكافأة دائمًا نبدأ من -inf
    start_step = 0
    total_episodes = 0

    # إنشاء مجلد النتائج
    try:
        os.makedirs(results_dir, exist_ok=True)
    except OSError as e:
        warnings.warn(f"Could not create results directory '{results_dir}': {e}. Saving disabled.", RuntimeWarning)
        save_best = False
        save_periodically = False

    # --- 3. محاولة المتابعة من نقطة تفتيش ---
    if resume_from_checkpoint:
        print(f"\n--- Resuming Training from Checkpoint: {resume_from_checkpoint} ---")
        if agent.load_model(resume_from_checkpoint):
            # تقدير الخطوة الحالية بناءً على التحديثات المحفوظة و update_frequency
            # (نفترض أن كل تحديث يتوافق مع update_frequency خطوة في البيئة)
            start_step = agent.total_updates # * update_frequency # <-- يجب أن يكون start_step هو عدد خطوات البيئة
            # إذا كان checkpoint يحفظ خطوة البيئة، استخدمها مباشرة.
            # هنا نفترض أن total_updates يتوافق مع عدد مرات استدعاء update().
            # نحتاج إلى تعديل كيفية حساب start_step إذا أردنا استئناف دقيق لخطوات البيئة.
            # للتبسيط، سنستأنف من عدد التحديثات المحفوظة. شريط التقدم سيبدأ من هنا.
            print(f"Resumed successfully. Resuming from update step: {agent.total_updates:,}")
            # قد تحتاج إلى استعادة best_eval_metric_value إذا تم حفظه في checkpoint
        else:
            warnings.warn("Failed to load checkpoint. Starting training from scratch.")
            start_step = 0 # البدء من الصفر
            agent.total_updates = 0 # إعادة تعيين عداد التحديثات

    # --- 4. إعادة تهيئة البيئة لبدء التدريب الفعلي ---
    try:
        # استخدام بذرة تعتمد على خطوة البدء لضمان استمرارية مختلفة إذا لم يتم الاستئناف
        state, info = env.reset(seed=RANDOM_SEED + start_step)
        state = np.asarray(state, dtype=np.float32) # ضمان النوع الصحيح
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to reset environment before training loop: {e}")
        env.close(); eval_env.close()
        return None, []

    # --- 5. حلقة التدريب الرئيسية ---
    print("\n--- Starting Training Loop ---")
    # شريط التقدم يعكس خطوات البيئة
    progress_bar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc="Training", unit="step", ncols=120) # زيادة عرض الشريط
    episode_reward = 0.0
    episode_steps = 0

    for current_step in progress_bar:
        # --- أ. التفاعل مع البيئة ---
        if current_step < start_learning_steps:
            # إجراء عشوائي في البداية لملء المخزن المؤقت
            action = env.action_space.sample()
        else:
            # إجراء من السياسة مع ضوضاء استكشاف
            action = agent.get_action(state, explore=True)

        try:
            # تنفيذ الإجراء في البيئة
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated # done = True إذا انتهت الحلقة لأي سبب

            # التعامل مع NaN/Inf المحتمل في المكافأة (من البيئة)
            if math.isnan(reward) or math.isinf(reward):
                 # لا تطبع تحذيرًا لكل خطوة لتجنب الإغراق
                 reward = 0.0 # استبدال بقيمة آمنة

            # تحويل الحالة التالية إلى النوع الصحيح
            next_state = np.asarray(next_state, dtype=np.float32)

            # تخزين التجربة في المخزن المؤقت
            # نستخدم float(done) لتخزين 0.0 أو 1.0
            agent.replay_buffer.push(state, action, reward, next_state, float(done))

            # تحديث الحالة الحالية
            state = next_state
            episode_reward += reward
            episode_steps += 1

            # --- ب. التعامل مع نهاية الحلقة (done = True) ---
            if done:
                total_episodes += 1
                # تحديث معلومات شريط التقدم باستخدام بيانات الحلقة المنتهية من wrapper
                if 'episode' in info:
                    # استخدام .get() لتجنب الخطأ إذا لم تكن المعلومات موجودة
                    ep_reward = info['episode'].get('r')
                    ep_length = info['episode'].get('l')
                    # حساب المتوسط المتحرك للمكافأة والطول من الـ deque في wrapper
                    avg_reward_recent = np.mean(env.return_queue) if env.return_queue else 0.0
                    avg_len_recent = np.mean(env.length_queue) if env.length_queue else 0.0

                    postfix_dict = {
                        "Ep": total_episodes,
                        "Rew(r)": f"{avg_reward_recent:.1f}", # متوسط المكافأة الأخير
                        "Len(r)": f"{avg_len_recent:.0f}", # متوسط الطول الأخير
                        "Noise": f"{agent.exploration_noise_std:.3f}",
                        "P_C": agent.policy_eq.complexity, # تعقيد السياسة
                        "Q_C": agent.q_eq.complexity     # تعقيد المقيّم
                    }
                    # إضافة الخسائر الأخيرة إذا كانت متاحة
                    if all_q_losses: postfix_dict["QL(r)"] = f"{np.mean(all_q_losses):.2f}"
                    if all_policy_losses: postfix_dict["PL(r)"] = f"{np.mean(all_policy_losses):.2f}"

                    progress_bar.set_postfix(postfix_dict)


                # إعادة تهيئة البيئة للحلقة التالية
                # استخدام بذرة تعتمد على الخطوة الحالية لضمان استمرارية مختلفة
                state, info = env.reset(seed=RANDOM_SEED + current_step)
                state = np.asarray(state, dtype=np.float32) # ضمان النوع الصحيح
                episode_reward = 0.0
                episode_steps = 0

        except (gym.error.Error, Exception) as e:
            warnings.warn(f"\nUnhandled exception during environment interaction at step {current_step}: {e}", RuntimeWarning)
            import traceback
            traceback.print_exc()
            print("Attempting to reset environment and continue...")
            try:
                 state, info = env.reset(seed=RANDOM_SEED + current_step + 1)
                 state = np.asarray(state, dtype=np.float32)
                 episode_reward = 0.0
                 episode_steps = 0
                 print("Environment reset successfully after error.")
            except Exception as e2:
                 print(f"CRITICAL ERROR: Failed to reset environment after interaction error: {e2}. Stopping training.")
                 break # إنهاء حلقة التدريب
            continue # الانتقال إلى الخطوة التالية

        # --- ج. تحديث الوكيل (الشبكات والتطور) ---
        if current_step >= start_learning_steps and current_step % update_frequency == 0:
            q_loss, policy_loss, batch_avg_reward = agent.update(step=agent.total_updates) # تمرير خطوة التحديث
            # تخزين الخسائر (تجاهل None إذا فشل التحديث)
            if q_loss is not None: all_q_losses.append(q_loss)
            if policy_loss is not None: all_policy_losses.append(policy_loss)

        # --- د. التقييم الدوري والحفظ ---
        # يتم التقييم بناءً على خطوات البيئة
        if current_step > 0 and current_step % eval_frequency == 0 and current_step >= start_learning_steps:
            # طباعة سطر فارغ قبل التقييم لتحسين القراءة عند استخدام tqdm
            progress_bar.write("\n" + "-"*40 + f" Evaluating at Step {current_step:,} " + "-"*40)
            # إجراء التقييم باستخدام بيئة التقييم وبدون استكشاف
            eval_avg_reward = agent.evaluate(eval_env, episodes=eval_episodes)
            evaluation_rewards.append(eval_avg_reward)
            steps_history.append(current_step) # تسجيل خطوة البيئة المقابلة للتقييم

            # استخدام progress_bar.write للطباعة المتوافقة مع tqdm
            progress_bar.write(f"  Avg Eval Reward ({eval_episodes} episodes): {eval_avg_reward:.2f}")
            progress_bar.write(f"  Agent Updates: {agent.total_updates:,} | P_Comp: {agent.policy_eq.complexity} ({agent.policy_struct_changes} changes) | Q_Comp: {agent.q_eq.complexity} ({agent.q_struct_changes} changes)")
            # طباعة متوسط الخسائر من الـ deque
            avg_q_loss_str = f"{np.mean(all_q_losses):.4f}" if all_q_losses else "N/A"
            avg_p_loss_str = f"{np.mean(all_policy_losses):.4f}" if all_policy_losses else "N/A"
            progress_bar.write(f"  Avg Q_Loss (recent): {avg_q_loss_str} | Avg P_Loss (recent): {avg_p_loss_str}")
            progress_bar.write("-" * 100)


            # --- حفظ أفضل نموذج بناءً على مكافأة التقييم ---
            if save_best:
                 # التحقق من أن المكافأة ليست لانهائية أو NaN قبل المقارنة
                 if not math.isinf(eval_avg_reward) and not math.isnan(eval_avg_reward) and eval_avg_reward > best_eval_metric_value:
                     old_best_str = f"{best_eval_metric_value:.2f}" if not math.isinf(best_eval_metric_value) else "-inf"
                     progress_bar.write(f"  ** New best eval reward ({eval_avg_reward:.2f} > {old_best_str})! Saving model... **")
                     best_eval_metric_value = eval_avg_reward
                     best_model_filename = os.path.join(results_dir, f"rle2_best_{env_name.replace('-','_')}.pt")
                     agent.save_model(filename=best_model_filename)
                 # إضافة شرط لحفظ النموذج الأول الذي يحقق أداءً لائقًا (اختياري)
                 # elif best_eval_metric_value == -np.inf and eval_avg_reward > -500: # مثال: أول مرة تتجاوز -500
                 #     progress_bar.write(f"  ** Saving first decent model (Reward: {eval_avg_reward:.2f})! **")
                 #     best_eval_metric_value = eval_avg_reward
                 #     best_model_filename = os.path.join(results_dir, f"rle2_best_{env_name.replace('-','_')}.pt")
                 #     agent.save_model(filename=best_model_filename)


        # --- هـ. الحفظ الدوري (بناءً على خطوات البيئة) ---
        if save_periodically and current_step > start_step and current_step % save_interval == 0:
             periodic_filename = os.path.join(results_dir, f"rle2_step_{current_step}_{env_name.replace('-','_')}.pt")
             progress_bar.write(f"\n--- Saving Periodic Checkpoint at Step {current_step:,} ---")
             agent.save_model(filename=periodic_filename)

    # --- 6. انتهاء التدريب والتنظيف والحفظ النهائي ---
    progress_bar.close()
    env.close()
    eval_env.close()

    # حفظ النموذج النهائي عند اكتمال التدريب
    final_model_filename = os.path.join(results_dir, f"rle2_final_step_{max_steps}_{env_name.replace('-','_')}.pt")
    print(f"\n--- Saving Final Model at Step {max_steps:,} ---")
    agent.save_model(filename=final_model_filename)


    end_time = time.time()
    total_training_time = end_time - start_time
    print("\n" + "="*60)
    print("=== Training Finished ===")
    print(f"=== Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print("="*60)
    print(f"Total Environment Steps: {max_steps:,}")
    print(f"Total Episodes Completed: {total_episodes:,}")
    print(f"Total Agent Updates: {agent.total_updates:,}")
    print(f"Total Training Time: {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)")
    if env.return_queue: print(f"Average reward of last {len(env.return_queue)} training episodes: {np.mean(env.return_queue):.2f}")
    print(f"Final Policy Complexity: {agent.policy_eq.complexity} (Total Changes: {agent.policy_struct_changes})")
    print(f"Final Q Complexity: {agent.q_eq.complexity} (Total Changes: {agent.q_struct_changes})")
    best_metric_str = f"{best_eval_metric_value:.2f}" if not math.isinf(best_eval_metric_value) else "N/A (or not saved)"
    print(f"Best Evaluation {save_best_metric.capitalize()} Achieved: {best_metric_str}")


    # --- 7. رسم بياني لنتائج التدريب ---
    if steps_history and evaluation_rewards:
        print("\n--- Generating Training Plots ---")
        try:
            # استخدام نمط رسم بياني متاح (تجنب seaborn إذا كان يسبب مشاكل)
            plt.style.use('ggplot') # بديل لـ seaborn
            fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            fig.suptitle(f'RL-E² Training Progress ({env_name} - DDPG-like)', fontsize=16)


            # --- رسم المكافأة ---
            ax1 = axes[0]
            ax1.plot(steps_history, evaluation_rewards, marker='.', linestyle='-', color='dodgerblue', label='Avg Eval Reward')
            # إضافة متوسط متحرك للمكافأة لتوضيح الاتجاه
            if len(evaluation_rewards) >= 5:
                 moving_avg = np.convolve(evaluation_rewards, np.ones(5)/5, mode='valid')
                 # محاذاة المتوسط المتحرك مع الخطوات الصحيحة (يبدأ من الخطوة الخامسة)
                 ax1.plot(steps_history[4:], moving_avg, linestyle='--', color='orangered', label='Moving Avg (5 evals)')

            ax1.set_ylabel('Average Evaluation Reward')
            ax1.set_title('Evaluation Reward Over Time')
            ax1.legend()
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

            # --- رسم خسارة المقيّم (Q-Loss) ---
            ax2 = axes[1]
            if all_q_losses:
                # تقدير خطوات التحديث المقابلة للخسائر (ليس دقيقًا تمامًا لكنه يعطي فكرة)
                update_steps_est = np.linspace(start_learning_steps, current_step, len(all_q_losses), dtype=int)
                ax2.plot(update_steps_est, list(all_q_losses), label='Q-Loss (Raw)', alpha=0.4, color='darkorange', linewidth=0.8)
                # حساب متوسط متحرك للخسارة لتنعيم الرسم البياني
                if len(all_q_losses) >= 20:
                    q_loss_ma = np.convolve(list(all_q_losses), np.ones(20)/20, mode='valid')
                    ax2.plot(update_steps_est[19:], q_loss_ma, label='Q-Loss (MA-20)', color='red', linewidth=1.2)

                ax2.set_ylabel('Q-Loss')
                ax2.set_title('Critic (Q-Function) Loss Over Time')
                # محاولة استخدام مقياس لوغاريتمي إذا كانت القيم موجبة وتختلف كثيرًا
                try:
                    if np.all(np.array(list(all_q_losses)) > 0):
                         ax2.set_yscale('log')
                         ax2.set_ylabel('Q-Loss (Log Scale)')
                except ValueError:
                    pass # تجاهل إذا فشل المقياس اللوغاريتمي
                ax2.legend()
                ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
            else:
                ax2.text(0.5, 0.5, 'No Q-Loss Data Recorded', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_ylabel('Q-Loss')

            # --- رسم خسارة السياسة (Policy Loss) ---
            ax3 = axes[2]
            if all_policy_losses:
                update_steps_est_p = np.linspace(start_learning_steps, current_step, len(all_policy_losses), dtype=int)
                ax3.plot(update_steps_est_p, list(all_policy_losses), label='Policy Loss (Raw)', alpha=0.4, color='forestgreen', linewidth=0.8)
                # حساب متوسط متحرك
                if len(all_policy_losses) >= 20:
                    p_loss_ma = np.convolve(list(all_policy_losses), np.ones(20)/20, mode='valid')
                    ax3.plot(update_steps_est_p[19:], p_loss_ma, label='Policy Loss (MA-20)', color='darkgreen', linewidth=1.2)

                ax3.set_ylabel('Policy Loss (-Avg Q)')
                ax3.set_title('Actor (Policy) Loss Over Time')
                ax3.legend()
                ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
            else:
                ax3.text(0.5, 0.5, 'No Policy Loss Data Recorded', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_ylabel('Policy Loss')

            ax3.set_xlabel('Environment Steps')
            fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # تعديل التخطيط لاستيعاب العنوان الرئيسي

            # حفظ الرسم البياني
            plot_filename = os.path.join(results_dir, f"rle2_training_plots_{env_name.replace('-','_')}.png")
            plt.savefig(plot_filename, dpi=300)
            print(f"Training plots saved to '{plot_filename}'")
            plt.close(fig) # إغلاق الشكل بعد الحفظ لتحرير الذاكرة

        except Exception as e:
            warnings.warn(f"Could not generate or save training plots: {e}", RuntimeWarning)
            import traceback
            traceback.print_exc()
    else:
        print("No evaluation data recorded (or evaluation frequency was too high), skipping plot generation.")

    return agent, evaluation_rewards


# --- 8. Execution Block ---
if __name__ == "__main__":

    print("\n" + "="*60)
    print("===          RL-E² Main Execution (v1.2 - DDPG-like)      ===")
    print("="*60)

    # --- تحديد البيئة والمعلمات ---
    ENVIRONMENT_NAME = "Pendulum-v1"
    # ENVIRONMENT_NAME = "MountainCarContinuous-v0" # قد تحتاج تعديل LR و noise وربما gamma
    # ENVIRONMENT_NAME = "LunarLanderContinuous-v2" # تحتاج خطوات أكثر (e.g., 500k+), تعقيد أكبر (e.g., 8-10), وقد تحتاج ضبط tau

    MAX_TRAINING_STEPS = 100000     # عدد خطوات مناسب لـ Pendulum
    EVALUATION_FREQUENCY = 5000     # تقييم كل 5000 خطوة بيئة
    START_LEARNING = 2500           # البدء في التحديث بعد جمع بعض التجارب الأولية
    BATCH_SIZE = 128                # حجم الدفعة للتدريب
    POLICY_LR = 5e-4                # معدل تعلم السياسة
    Q_LR = 5e-4                     # معدل تعلم المقيّم
    TAU = 0.005                     # معامل التحديث الناعم للهدف
    EXPLORATION_NOISE = 0.4         # ضوضاء استكشاف أولية أعلى قليلاً
    NOISE_DECAY = 0.9999            # تناقص أبطأ للضوضاء (9999 بدلاً من 9998)
    MIN_NOISE = 0.1                 # الحد الأدنى للضوضاء أعلى قليلاً

    SAVE_BEST_MODEL = True          # حفظ أفضل نموذج بناءً على مكافأة التقييم
    SAVE_PERIODICALLY = True        # حفظ نقاط تفتيش دورية
    SAVE_INTERVAL = 25000           # حفظ كل 25000 خطوة بيئة

    # اسم مجلد فريد لكل تجربة
    RESULTS_DIRECTORY = f"rle2_results_{ENVIRONMENT_NAME.replace('-','_')}_{time.strftime('%Y%m%d_%H%M%S')}"

    # استئناف التدريب من نقطة تفتيش (ضع المسار هنا أو اتركه None)
    RESUME_CHECKPOINT = None # "path/to/your/rle2_best_Pendulum_v1.pt"

    # --- بدء التدريب ---
    trained_agent, eval_history = train_rle2(
        env_name=ENVIRONMENT_NAME,
        max_steps=MAX_TRAINING_STEPS,
        batch_size=BATCH_SIZE,
        eval_frequency=EVALUATION_FREQUENCY,
        start_learning_steps=START_LEARNING,
        policy_lr=POLICY_LR,
        q_lr=Q_LR,
        tau=TAU,
        exploration_noise=EXPLORATION_NOISE,
        noise_decay=NOISE_DECAY,
        min_noise=MIN_NOISE,
        save_best=SAVE_BEST_MODEL,
        save_best_metric='reward', # المقياس المستخدم لتحديد "الأفضل"
        save_periodically=SAVE_PERIODICALLY,
        save_interval=SAVE_INTERVAL,
        results_dir=RESULTS_DIRECTORY,
        resume_from_checkpoint=RESUME_CHECKPOINT,
        render_eval=False, # تعطيل العرض أثناء التقييم الدوري (أسرع)
        eval_episodes=5    # عدد الحلقات لكل تقييم
    )

    # --- التقييم النهائي بعد انتهاء التدريب ---
    if trained_agent:
        print("\n" + "="*60)
        print("=== Final Evaluation of Trained Agent ===")
        print("="*60)
        try:
            # محاولة التقييم مع العرض المرئي (يتطلب بيئة تدعم 'human' render_mode)
            print("Evaluating final agent with rendering (if available)...")
            final_eval_env = gym.make(ENVIRONMENT_NAME, render_mode="human")
            final_performance = trained_agent.evaluate(final_eval_env, episodes=10)
            print(f"Final Agent Avg Performance (10 episodes, rendered): {final_performance:.2f}")
            final_eval_env.close()
        except Exception as e:
            print(f"  * Could not run evaluation with rendering: {e}")
            print("  * Evaluating final agent without rendering...")
            try:
                 final_eval_env_no_render = gym.make(ENVIRONMENT_NAME)
                 final_performance = trained_agent.evaluate(final_eval_env_no_render, episodes=10)
                 print(f"  * Final Agent Avg Performance (10 episodes, no render): {final_performance:.2f}")
                 final_eval_env_no_render.close()
            except Exception as e2:
                 print(f"  * ERROR: Failed final evaluation without rendering: {e2}")

        # --- تقييم أفضل نموذج تم حفظه (إذا تم الحفظ) ---
        if SAVE_BEST_MODEL and os.path.exists(RESULTS_DIRECTORY):
            best_model_filename = os.path.join(RESULTS_DIRECTORY, f"rle2_best_{ENVIRONMENT_NAME.replace('-','_')}.pt")
            print("\n" + "="*60)
            print(f"=== Evaluating Best Saved Agent ({os.path.basename(best_model_filename)}) ===")
            print("="*60)
            if os.path.exists(best_model_filename):
                try:
                    # إنشاء وكيل جديد لتحميل النموذج الأفضل
                    # استخدام نفس الأبعاد والحدود من الوكيل المدرب
                    eval_best_agent = RLE2Agent(trained_agent.state_dim, trained_agent.action_dim,
                                            (trained_agent.action_low, trained_agent.action_high))

                    if eval_best_agent.load_model(best_model_filename):
                        print("Best model loaded. Evaluating...")
                        best_eval_env = gym.make(ENVIRONMENT_NAME)
                        best_performance = eval_best_agent.evaluate(best_eval_env, episodes=10)
                        print(f"Best Saved Agent Avg Performance (10 episodes): {best_performance:.2f}")
                        print(f"  Policy Complexity: {eval_best_agent.policy_eq.complexity}")
                        print(f"  Q Complexity: {eval_best_agent.q_eq.complexity}")
                        best_eval_env.close()
                    else:
                        print("Skipping evaluation of best model (load failed).")

                except Exception as e:
                    print(f"ERROR during loading/evaluating best model: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Best model file not found at '{best_model_filename}'. Skipping evaluation.")
    else:
        print("\nTraining failed, was interrupted, or agent initialization failed. No final agent available.")

    print("\n===================================================")
    print("===            RL-E² Execution End            ===")
    print("===================================================")

