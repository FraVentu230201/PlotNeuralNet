diff --git a//dev/null b/pyexamples/resnet.py
index 0000000000000000000000000000000000000000..496fc23149c93d35bbe6f8c7aaa84d12554d289b 100644
--- a//dev/null
+++ b/pyexamples/resnet.py
@@ -0,0 +1,74 @@
+import os
+import sys
+
+sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
+from pycore.tikzeng import *
+
+
+# ResNet-style architecture with multiple residual blocks
+arch = [
+    to_head('../'),
+    to_cor(),
+    to_begin(),
+
+    to_input('../examples/fcn8s/cats.jpg', width=8, height=8, name='input'),
+
+    to_Conv('conv1', 3, 64, offset="(1,0,0)", to="(0,0,0)",
+            height=64, depth=64, width=2, caption='conv1'),
+    to_Pool('pool1', offset="(0,0,0)", to="(conv1-east)",
+            height=32, depth=32, width=1, caption='pool1'),
+
+    # Residual block 1
+    to_Conv('res2_a', 64, 64, offset="(1,0,0)", to="(pool1-east)",
+            height=32, depth=32, width=2),
+    to_Conv('res2_b', 64, 64, offset="(0,0,0)", to="(res2_a-east)",
+            height=32, depth=32, width=2),
+    to_Sum('res2_sum', offset="(0.5,0,0)", to="(res2_b-east)"),
+    to_connection('res2_b', 'res2_sum'),
+    to_skip('pool1', 'res2_sum'),
+
+    # Residual block 2
+    to_Conv('res3_a', 128, 128, offset="(1,0,0)", to="(res2_sum-east)",
+            height=28, depth=28, width=2),
+    to_Conv('res3_b', 128, 128, offset="(0,0,0)", to="(res3_a-east)",
+            height=28, depth=28, width=2),
+    to_Sum('res3_sum', offset="(0.5,0,0)", to="(res3_b-east)"),
+    to_connection('res3_b', 'res3_sum'),
+    to_skip('res2_sum', 'res3_sum'),
+
+    # Residual block 3
+    to_Conv('res4_a', 256, 256, offset="(1,0,0)", to="(res3_sum-east)",
+            height=20, depth=20, width=2),
+    to_Conv('res4_b', 256, 256, offset="(0,0,0)", to="(res4_a-east)",
+            height=20, depth=20, width=2),
+    to_Sum('res4_sum', offset="(0.5,0,0)", to="(res4_b-east)"),
+    to_connection('res4_b', 'res4_sum'),
+    to_skip('res3_sum', 'res4_sum'),
+
+    # Residual block 4
+    to_Conv('res5_a', 512, 512, offset="(1,0,0)", to="(res4_sum-east)",
+            height=12, depth=12, width=2),
+    to_Conv('res5_b', 512, 512, offset="(0,0,0)", to="(res5_a-east)",
+            height=12, depth=12, width=2),
+    to_Sum('res5_sum', offset="(0.5,0,0)", to="(res5_b-east)"),
+    to_connection('res5_b', 'res5_sum'),
+    to_skip('res4_sum', 'res5_sum'),
+
+    to_Pool('avgpool', offset="(1,0,0)", to="(res5_sum-east)",
+            height=4, depth=4, width=1, caption='avgpool'),
+    to_Conv('fc', 512, 1000, offset="(1,0,0)", to="(avgpool-east)",
+            height=4, depth=4, width=1, caption='fc'),
+    to_SoftMax('softmax', 1000, "(1,0,0)", "(fc-east)", caption='softmax'),
+
+    to_end()
+]
+
+
+def main():
+    namefile = os.path.splitext(os.path.basename(__file__))[0]
+    to_generate(arch, namefile + '.tex')
+
+
+if __name__ == '__main__':
+    main()
+
