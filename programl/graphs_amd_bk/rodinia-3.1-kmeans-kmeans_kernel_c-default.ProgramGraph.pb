

[external]
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 0) #3
5truncB,
*
	full_text

%10 = trunc i64 %9 to i32
"i64B

	full_text


i64 %9
5icmpB-
+
	full_text

%11 = icmp ult i32 %10, %3
#i32B

	full_text
	
i32 %10
8brB2
0
	full_text#
!
br i1 %11, label %12, label %86
!i1B

	full_text


i1 %11
5icmp8B+
)
	full_text

%13 = icmp sgt i32 %4, 0
:br8B2
0
	full_text#
!
br i1 %13, label %14, label %24
#i18B

	full_text


i1 %13
5icmp8B+
)
	full_text

%15 = icmp sgt i32 %5, 0
5sext8B+
)
	full_text

%16 = sext i32 %3 to i64
5sext8B+
)
	full_text

%17 = sext i32 %5 to i64
5zext8B+
)
	full_text

%18 = zext i32 %5 to i64
5zext8B+
)
	full_text

%19 = zext i32 %4 to i64
0and8B'
%
	full_text

%20 = and i64 %18, 1
%i648B

	full_text
	
i64 %18
4icmp8B*
(
	full_text

%21 = icmp eq i32 %5, 1
6sub8B-
+
	full_text

%22 = sub nsw i64 %18, %20
%i648B

	full_text
	
i64 %18
%i648B

	full_text
	
i64 %20
5icmp8B+
)
	full_text

%23 = icmp eq i64 %20, 0
%i648B

	full_text
	
i64 %20
'br8B

	full_text

br label %28
Bphi8B9
7
	full_text*
(
&%25 = phi i32 [ 0, %12 ], [ %54, %50 ]
%i328B

	full_text
	
i32 %54
8and8B/
-
	full_text 

%26 = and i64 %9, 4294967295
$i648B

	full_text


i64 %9
Xgetelementptr8BE
C
	full_text6
4
2%27 = getelementptr inbounds i32, i32* %2, i64 %26
%i648B

	full_text
	
i64 %26
Hstore8B=
;
	full_text.
,
*store i32 %25, i32* %27, align 4, !tbaa !8
%i328B

	full_text
	
i32 %25
'i32*8B

	full_text


i32* %27
'br8B

	full_text

br label %86
Bphi8B9
7
	full_text*
(
&%29 = phi i64 [ 0, %14 ], [ %56, %50 ]
%i648B

	full_text
	
i64 %56
Uphi8BL
J
	full_text=
;
9%30 = phi float [ 0x47EFFFFFE0000000, %14 ], [ %55, %50 ]
)float8B

	full_text

	float %55
Bphi8B9
7
	full_text*
(
&%31 = phi i32 [ 0, %14 ], [ %54, %50 ]
%i328B

	full_text
	
i32 %54
:br8B2
0
	full_text#
!
br i1 %15, label %32, label %50
#i18B

	full_text


i1 %15
6mul8B-
+
	full_text

%33 = mul nsw i64 %29, %17
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %17
:br8B2
0
	full_text#
!
br i1 %21, label %35, label %34
#i18B

	full_text


i1 %21
'br8B

	full_text

br label %58
Hphi8B?
=
	full_text0
.
,%36 = phi float [ undef, %32 ], [ %82, %58 ]
)float8B

	full_text

	float %82
Bphi8B9
7
	full_text*
(
&%37 = phi i64 [ 0, %32 ], [ %83, %58 ]
%i648B

	full_text
	
i64 %83
Ophi8BF
D
	full_text7
5
3%38 = phi float [ 0.000000e+00, %32 ], [ %82, %58 ]
)float8B

	full_text

	float %82
:br8B2
0
	full_text#
!
br i1 %23, label %50, label %39
#i18B

	full_text


i1 %23
6mul8B-
+
	full_text

%40 = mul nsw i64 %37, %16
%i648B

	full_text
	
i64 %37
%i648B

	full_text
	
i64 %16
1add8B(
&
	full_text

%41 = add i64 %40, %9
%i648B

	full_text
	
i64 %40
$i648B

	full_text


i64 %9
9and8B0
.
	full_text!

%42 = and i64 %41, 4294967295
%i648B

	full_text
	
i64 %41
\getelementptr8BI
G
	full_text:
8
6%43 = getelementptr inbounds float, float* %0, i64 %42
%i648B

	full_text
	
i64 %42
Mload8BC
A
	full_text4
2
0%44 = load float, float* %43, align 4, !tbaa !12
+float*8B

	full_text


float* %43
6add8B-
+
	full_text

%45 = add nsw i64 %37, %33
%i648B

	full_text
	
i64 %37
%i648B

	full_text
	
i64 %33
\getelementptr8BI
G
	full_text:
8
6%46 = getelementptr inbounds float, float* %1, i64 %45
%i648B

	full_text
	
i64 %45
Mload8BC
A
	full_text4
2
0%47 = load float, float* %46, align 4, !tbaa !12
+float*8B

	full_text


float* %46
6fsub8B,
*
	full_text

%48 = fsub float %44, %47
)float8B

	full_text

	float %44
)float8B

	full_text

	float %47
ecall8B[
Y
	full_textL
J
H%49 = tail call float @llvm.fmuladd.f32(float %48, float %48, float %38)
)float8B

	full_text

	float %48
)float8B

	full_text

	float %48
)float8B

	full_text

	float %38
'br8B

	full_text

br label %50
]phi8	BT
R
	full_textE
C
A%51 = phi float [ 0.000000e+00, %28 ], [ %36, %35 ], [ %49, %39 ]
)float8	B

	full_text

	float %36
)float8	B

	full_text

	float %49
:fcmp8	B0
.
	full_text!

%52 = fcmp olt float %51, %30
)float8	B

	full_text

	float %51
)float8	B

	full_text

	float %30
8trunc8	B-
+
	full_text

%53 = trunc i64 %29 to i32
%i648	B

	full_text
	
i64 %29
Dselect8	B8
6
	full_text)
'
%%54 = select i1 %52, i32 %53, i32 %31
#i18	B

	full_text


i1 %52
%i328	B

	full_text
	
i32 %53
%i328	B

	full_text
	
i32 %31
Hselect8	B<
:
	full_text-
+
)%55 = select i1 %52, float %51, float %30
#i18	B

	full_text


i1 %52
)float8	B

	full_text

	float %51
)float8	B

	full_text

	float %30
8add8	B/
-
	full_text 

%56 = add nuw nsw i64 %29, 1
%i648	B

	full_text
	
i64 %29
7icmp8	B-
+
	full_text

%57 = icmp eq i64 %56, %19
%i648	B

	full_text
	
i64 %56
%i648	B

	full_text
	
i64 %19
:br8	B2
0
	full_text#
!
br i1 %57, label %24, label %28
#i18	B

	full_text


i1 %57
Bphi8
B9
7
	full_text*
(
&%59 = phi i64 [ 0, %34 ], [ %83, %58 ]
%i648
B

	full_text
	
i64 %83
Ophi8
BF
D
	full_text7
5
3%60 = phi float [ 0.000000e+00, %34 ], [ %82, %58 ]
)float8
B

	full_text

	float %82
Dphi8
B;
9
	full_text,
*
(%61 = phi i64 [ %22, %34 ], [ %84, %58 ]
%i648
B

	full_text
	
i64 %22
%i648
B

	full_text
	
i64 %84
6mul8
B-
+
	full_text

%62 = mul nsw i64 %59, %16
%i648
B

	full_text
	
i64 %59
%i648
B

	full_text
	
i64 %16
1add8
B(
&
	full_text

%63 = add i64 %62, %9
%i648
B

	full_text
	
i64 %62
$i648
B

	full_text


i64 %9
9and8
B0
.
	full_text!

%64 = and i64 %63, 4294967295
%i648
B

	full_text
	
i64 %63
\getelementptr8
BI
G
	full_text:
8
6%65 = getelementptr inbounds float, float* %0, i64 %64
%i648
B

	full_text
	
i64 %64
Mload8
BC
A
	full_text4
2
0%66 = load float, float* %65, align 4, !tbaa !12
+float*8
B

	full_text


float* %65
6add8
B-
+
	full_text

%67 = add nsw i64 %59, %33
%i648
B

	full_text
	
i64 %59
%i648
B

	full_text
	
i64 %33
\getelementptr8
BI
G
	full_text:
8
6%68 = getelementptr inbounds float, float* %1, i64 %67
%i648
B

	full_text
	
i64 %67
Mload8
BC
A
	full_text4
2
0%69 = load float, float* %68, align 4, !tbaa !12
+float*8
B

	full_text


float* %68
6fsub8
B,
*
	full_text

%70 = fsub float %66, %69
)float8
B

	full_text

	float %66
)float8
B

	full_text

	float %69
ecall8
B[
Y
	full_textL
J
H%71 = tail call float @llvm.fmuladd.f32(float %70, float %70, float %60)
)float8
B

	full_text

	float %70
)float8
B

	full_text

	float %70
)float8
B

	full_text

	float %60
.or8
B&
$
	full_text

%72 = or i64 %59, 1
%i648
B

	full_text
	
i64 %59
6mul8
B-
+
	full_text

%73 = mul nsw i64 %72, %16
%i648
B

	full_text
	
i64 %72
%i648
B

	full_text
	
i64 %16
1add8
B(
&
	full_text

%74 = add i64 %73, %9
%i648
B

	full_text
	
i64 %73
$i648
B

	full_text


i64 %9
9and8
B0
.
	full_text!

%75 = and i64 %74, 4294967295
%i648
B

	full_text
	
i64 %74
\getelementptr8
BI
G
	full_text:
8
6%76 = getelementptr inbounds float, float* %0, i64 %75
%i648
B

	full_text
	
i64 %75
Mload8
BC
A
	full_text4
2
0%77 = load float, float* %76, align 4, !tbaa !12
+float*8
B

	full_text


float* %76
6add8
B-
+
	full_text

%78 = add nsw i64 %72, %33
%i648
B

	full_text
	
i64 %72
%i648
B

	full_text
	
i64 %33
\getelementptr8
BI
G
	full_text:
8
6%79 = getelementptr inbounds float, float* %1, i64 %78
%i648
B

	full_text
	
i64 %78
Mload8
BC
A
	full_text4
2
0%80 = load float, float* %79, align 4, !tbaa !12
+float*8
B

	full_text


float* %79
6fsub8
B,
*
	full_text

%81 = fsub float %77, %80
)float8
B

	full_text

	float %77
)float8
B

	full_text

	float %80
ecall8
B[
Y
	full_textL
J
H%82 = tail call float @llvm.fmuladd.f32(float %81, float %81, float %71)
)float8
B

	full_text

	float %81
)float8
B

	full_text

	float %81
)float8
B

	full_text

	float %71
4add8
B+
)
	full_text

%83 = add nsw i64 %59, 2
%i648
B

	full_text
	
i64 %59
1add8
B(
&
	full_text

%84 = add i64 %61, -2
%i648
B

	full_text
	
i64 %61
5icmp8
B+
)
	full_text

%85 = icmp eq i64 %84, 0
%i648
B

	full_text
	
i64 %84
:br8
B2
0
	full_text#
!
br i1 %85, label %35, label %58
#i18
B

	full_text


i1 %85
$ret8B

	full_text


ret void
&i32*8B

	full_text
	
i32* %2
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %4
*float*8B

	full_text

	float* %1
$i328B

	full_text


i32 %5
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 0
,i648B!

	full_text

i64 4294967295
#i328B

	full_text	

i32 1
8float8B+
)
	full_text

float 0x47EFFFFFE0000000
+float8B

	full_text

float undef
$i648B

	full_text


i64 -2
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 2
2float8B%
#
	full_text

float 0.000000e+00       	
 	                     !  "$ ## %& %% '( '' )* ), +- ++ ./ .2 11 34 33 56 55 78 7: 9; 99 <= <> << ?@ ?? AB AA CD CC EF EG EE HI HH JK JJ LM LN LL OP OQ OR OO SU TV TT WX WY WW Z[ ZZ \] \^ \_ \\ `a `b `c `` de dd fg fh ff ij il kk mn mm op oq oo rs rt rr uv uw uu xy xx z{ zz |} || ~ ~	€ ~~ 
‚  ƒ„ ƒƒ …† …
‡ …… ˆ‰ ˆ
Š ˆ
‹ ˆˆ Œ ŒŒ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”• ”” –
— –– ˜™ ˜˜ š› š
œ šš 
ž  Ÿ  ŸŸ ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤
§ ¤¤ ¨© ¨¨ ª« ªª ¬­ ¬¬ ®¯ ®± 	² ² ³ A³ z³ –´ ´ µ Hµ µ ¶ ¶ ¶ ¶     
    \      !d $` &\ ( *# , - /¤ 2¨ 4¤ 6 83 : ;9 = >< @? BA D3 F+ GE IH KC MJ NL PL Q5 R1 UO VT X% Y# [W ]Z ^' _W aT b% c# ed g hf j¨ l¤ n pª qk s tr v wu yx {z }k + €~ ‚ „| †ƒ ‡… ‰… Šm ‹k Œ  Ž ’ “‘ •” —– ™Œ ›+ œš ž  ˜ ¢Ÿ £¡ ¥¡ ¦ˆ §k ©o «ª ­¬ ¯  °	 	  #" °) +) T. 1. 0i i #7 T7 90 kS T® 1® k ·· ¸¸ ° ·· O ¸¸ Oˆ ¸¸ ˆ¤ ¸¸ ¤¹ 	¹ 	¹ ¹ ¹ '	º 	º ?	º x
º ”	» ¼ %½ 1
¾ ª	¿ 	¿ d
¿ Œ	À À #À 3À k
À ¬
Á ¨Â 5Â TÂ m"
kmeans_kernel_c"
_Z13get_global_idj"
llvm.fmuladd.f32*ž
%rodinia-3.1-kmeans-kmeans_kernel_c.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02
 
transfer_bytes_log1p
Ø•A

wgsize_log1p
Ø•A

devmap_label


wgsize
€

transfer_bytes
ø“‚A