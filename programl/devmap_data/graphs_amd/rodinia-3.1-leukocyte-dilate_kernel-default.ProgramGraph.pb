

[external]
.sdivB&
$
	full_text

%8 = sdiv i32 %2, 2
.sdivB&
$
	full_text

%9 = sdiv i32 %3, 2
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 0) #2
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
1sremB)
'
	full_text

%12 = srem i32 %11, %0
#i32B

	full_text
	
i32 %11
1sdivB)
'
	full_text

%13 = sdiv i32 %11, %0
#i32B

	full_text
	
i32 %11
5icmpB-
+
	full_text

%14 = icmp slt i32 %13, %1
#i32B

	full_text
	
i32 %13
8brB2
0
	full_text#
!
br i1 %14, label %15, label %71
!i1B

	full_text


i1 %14
5icmp8B+
)
	full_text

%16 = icmp sgt i32 %2, 0
:br8B2
0
	full_text#
!
br i1 %16, label %17, label %65
#i18B

	full_text


i1 %16
5sub8B,
*
	full_text

%18 = sub nsw i32 %12, %8
%i328B

	full_text
	
i32 %12
$i328B

	full_text


i32 %8
5icmp8B+
)
	full_text

%19 = icmp slt i32 %3, 1
5sub8B,
*
	full_text

%20 = sub nsw i32 %13, %9
%i328B

	full_text
	
i32 %13
$i328B

	full_text


i32 %9
6sext8B,
*
	full_text

%21 = sext i32 %20 to i64
%i328B

	full_text
	
i32 %20
5sext8B+
)
	full_text

%22 = sext i32 %1 to i64
5sext8B+
)
	full_text

%23 = sext i32 %0 to i64
5sext8B+
)
	full_text

%24 = sext i32 %3 to i64
6sext8B,
*
	full_text

%25 = sext i32 %18 to i64
%i328B

	full_text
	
i32 %18
5zext8B+
)
	full_text

%26 = zext i32 %2 to i64
5zext8B+
)
	full_text

%27 = zext i32 %3 to i64
'br8B

	full_text

br label %28
Bphi8B9
7
	full_text*
(
&%29 = phi i64 [ 0, %17 ], [ %63, %61 ]
%i648B

	full_text
	
i64 %63
Ophi8BF
D
	full_text7
5
3%30 = phi float [ 0.000000e+00, %17 ], [ %62, %61 ]
)float8B

	full_text

	float %62
6add8B-
+
	full_text

%31 = add nsw i64 %29, %25
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %25
6icmp8B,
*
	full_text

%32 = icmp slt i64 %31, 0
%i648B

	full_text
	
i64 %31
8icmp8B.
,
	full_text

%33 = icmp sge i64 %31, %23
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %23
/or8B'
%
	full_text

%34 = or i1 %33, %32
#i18B

	full_text


i1 %33
#i18B

	full_text


i1 %32
/or8B'
%
	full_text

%35 = or i1 %34, %19
#i18B

	full_text


i1 %34
#i18B

	full_text


i1 %19
:br8B2
0
	full_text#
!
br i1 %35, label %61, label %36
#i18B

	full_text


i1 %35
6mul8B-
+
	full_text

%37 = mul nsw i64 %29, %24
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %24
'br8B

	full_text

br label %38
Bphi8B9
7
	full_text*
(
&%39 = phi i64 [ 0, %36 ], [ %59, %57 ]
%i648B

	full_text
	
i64 %59
Fphi8B=
;
	full_text.
,
*%40 = phi float [ %30, %36 ], [ %58, %57 ]
)float8B

	full_text

	float %30
)float8B

	full_text

	float %58
6add8B-
+
	full_text

%41 = add nsw i64 %39, %21
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %21
7icmp8B-
+
	full_text

%42 = icmp sgt i64 %41, -1
%i648B

	full_text
	
i64 %41
8icmp8B.
,
	full_text

%43 = icmp slt i64 %41, %22
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %22
1and8B(
&
	full_text

%44 = and i1 %42, %43
#i18B

	full_text


i1 %42
#i18B

	full_text


i1 %43
:br8B2
0
	full_text#
!
br i1 %44, label %45, label %57
#i18B

	full_text


i1 %44
6add8B-
+
	full_text

%46 = add nsw i64 %39, %37
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%47 = getelementptr inbounds float, float* %4, i64 %46
%i648B

	full_text
	
i64 %46
Lload8BB
@
	full_text3
1
/%48 = load float, float* %47, align 4, !tbaa !8
+float*8B

	full_text


float* %47
Cfcmp8B9
7
	full_text*
(
&%49 = fcmp une float %48, 0.000000e+00
)float8B

	full_text

	float %48
:br8B2
0
	full_text#
!
br i1 %49, label %50, label %57
#i18B

	full_text


i1 %49
6mul8B-
+
	full_text

%51 = mul nsw i64 %41, %23
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %23
6add8B-
+
	full_text

%52 = add nsw i64 %51, %31
%i648B

	full_text
	
i64 %51
%i648B

	full_text
	
i64 %31
\getelementptr8BI
G
	full_text:
8
6%53 = getelementptr inbounds float, float* %5, i64 %52
%i648B

	full_text
	
i64 %52
Lload8BB
@
	full_text3
1
/%54 = load float, float* %53, align 4, !tbaa !8
+float*8B

	full_text


float* %53
:fcmp8B0
.
	full_text!

%55 = fcmp ogt float %54, %40
)float8B

	full_text

	float %54
)float8B

	full_text

	float %40
Hselect8B<
:
	full_text-
+
)%56 = select i1 %55, float %54, float %40
#i18B

	full_text


i1 %55
)float8B

	full_text

	float %54
)float8B

	full_text

	float %40
'br8B

	full_text

br label %57
Tphi8BK
I
	full_text<
:
8%58 = phi float [ %40, %45 ], [ %40, %38 ], [ %56, %50 ]
)float8B

	full_text

	float %40
)float8B

	full_text

	float %40
)float8B

	full_text

	float %56
8add8B/
-
	full_text 

%59 = add nuw nsw i64 %39, 1
%i648B

	full_text
	
i64 %39
7icmp8B-
+
	full_text

%60 = icmp eq i64 %59, %27
%i648B

	full_text
	
i64 %59
%i648B

	full_text
	
i64 %27
:br8B2
0
	full_text#
!
br i1 %60, label %61, label %38
#i18B

	full_text


i1 %60
Fphi8	B=
;
	full_text.
,
*%62 = phi float [ %30, %28 ], [ %58, %57 ]
)float8	B

	full_text

	float %30
)float8	B

	full_text

	float %58
8add8	B/
-
	full_text 

%63 = add nuw nsw i64 %29, 1
%i648	B

	full_text
	
i64 %29
7icmp8	B-
+
	full_text

%64 = icmp eq i64 %63, %26
%i648	B

	full_text
	
i64 %63
%i648	B

	full_text
	
i64 %26
:br8	B2
0
	full_text#
!
br i1 %64, label %65, label %28
#i18	B

	full_text


i1 %64
Ophi8
BF
D
	full_text7
5
3%66 = phi float [ 0.000000e+00, %15 ], [ %62, %61 ]
)float8
B

	full_text

	float %62
5mul8
B,
*
	full_text

%67 = mul nsw i32 %12, %1
%i328
B

	full_text
	
i32 %12
6add8
B-
+
	full_text

%68 = add nsw i32 %67, %13
%i328
B

	full_text
	
i32 %67
%i328
B

	full_text
	
i32 %13
6sext8
B,
*
	full_text

%69 = sext i32 %68 to i64
%i328
B

	full_text
	
i32 %68
\getelementptr8
BI
G
	full_text:
8
6%70 = getelementptr inbounds float, float* %6, i64 %69
%i648
B

	full_text
	
i64 %69
Lstore8
BA
?
	full_text2
0
.store float %66, float* %70, align 4, !tbaa !8
)float8
B

	full_text

	float %66
+float*8
B

	full_text


float* %70
'br8
B

	full_text

br label %71
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %2
*float*8B

	full_text

	float* %6
$i328B

	full_text


i32 %1
$i328B

	full_text


i32 %0
*float*8B

	full_text

	float* %5
*float*8B

	full_text

	float* %4
$i328B

	full_text


i32 %3
-; undefined function B

	full_text

 
$i648B

	full_text


i64 -1
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 1
2float8B%
#
	full_text

float 0.000000e+00
#i328B

	full_text	

i32 2       	  
 

                      !# "" $% $$ &' &( && )* )) +, +- ++ ./ .0 .. 12 13 11 45 47 68 66 9; :: <= <> << ?@ ?A ?? BC BB DE DF DD GH GI GG JK JM LN LL OP OO QR QQ ST SS UV UX WY WW Z[ Z\ ZZ ]^ ]] _` __ ab ac aa de df dg dd hj ik il ii mn mm op oq oo rs ru tv tt wx ww yz y{ yy |} | ~~ ?? ?? ?? ?
? ?? ?? ?? ?
? ?? ?? ?
? ?? ?? ? ? ? ?	? 
? 
? ?	? 	? ? ? ]? O? ? ? ?     	 
        w #t %" ' (& *& , -+ /) 0. 2 31 5" 7 8m ;$ =i >: @ A? C? E FB HD IG K: M6 NL PO RQ TS V? X YW [& \Z ^] `_ b< ca e_ f< g< j< kd l: nm p  qo s$ ui v" xw z {y }t  ?? ? ?? ?? ?~ ?? ?  ?  ~! "? ?4 t4 6| ~| "9 :J LJ iU WU ir tr :h i ? ?? ?? 	? B	? m	? w? 	? ? "	? )? :	? ? $	? S? ~	? 	? "
dilate_kernel"
_Z13get_global_idj*?
&rodinia-3.1-leukocyte-dilate_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?
 
transfer_bytes_log1p
G2?A

wgsize
?

devmap_label


wgsize_log1p
G2?A

transfer_bytes
???