
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

%11 = icmp slt i32 %10, %7
#i32B

	full_text
	
i32 %10
8brB2
0
	full_text#
!
br i1 %11, label %12, label %46
!i1B

	full_text


i1 %11
5icmp8B+
)
	full_text

%13 = icmp sgt i32 %7, 0
0shl8B'
%
	full_text

%14 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%15 = ashr exact i64 %14, 32
%i648B

	full_text
	
i64 %14
:br8B2
0
	full_text#
!
br i1 %13, label %16, label %31
#i18B

	full_text


i1 %13
\getelementptr8BI
G
	full_text:
8
6%17 = getelementptr inbounds float, float* %3, i64 %15
%i648B

	full_text
	
i64 %15
Lload8BB
@
	full_text3
1
/%18 = load float, float* %17, align 4, !tbaa !8
+float*8B

	full_text


float* %17
5sext8B+
)
	full_text

%19 = sext i32 %7 to i64
'br8B

	full_text

br label %20
Bphi8B9
7
	full_text*
(
&%21 = phi i64 [ 0, %16 ], [ %26, %25 ]
%i648B

	full_text
	
i64 %26
\getelementptr8BI
G
	full_text:
8
6%22 = getelementptr inbounds float, float* %2, i64 %21
%i648B

	full_text
	
i64 %21
Lload8BB
@
	full_text3
1
/%23 = load float, float* %22, align 4, !tbaa !8
+float*8B

	full_text


float* %22
:fcmp8B0
.
	full_text!

%24 = fcmp ult float %23, %18
)float8B

	full_text

	float %23
)float8B

	full_text

	float %18
:br8B2
0
	full_text#
!
br i1 %24, label %25, label %28
#i18B

	full_text


i1 %24
8add8B/
-
	full_text 

%26 = add nuw nsw i64 %21, 1
%i648B

	full_text
	
i64 %21
8icmp8B.
,
	full_text

%27 = icmp slt i64 %26, %19
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %19
:br8B2
0
	full_text#
!
br i1 %27, label %20, label %31
#i18B

	full_text


i1 %27
8trunc8B-
+
	full_text

%29 = trunc i64 %21 to i32
%i648B

	full_text
	
i64 %21
6icmp8B,
*
	full_text

%30 = icmp eq i32 %29, -1
%i328B

	full_text
	
i32 %29
:br8B2
0
	full_text#
!
br i1 %30, label %31, label %33
#i18B

	full_text


i1 %30
4add8B+
)
	full_text

%32 = add nsw i32 %7, -1
'br8B

	full_text

br label %33
Dphi8B;
9
	full_text,
*
(%34 = phi i32 [ %29, %28 ], [ %32, %31 ]
%i328B

	full_text
	
i32 %29
%i328B

	full_text
	
i32 %32
6sext8B,
*
	full_text

%35 = sext i32 %34 to i64
%i328B

	full_text
	
i32 %34
\getelementptr8BI
G
	full_text:
8
6%36 = getelementptr inbounds float, float* %0, i64 %35
%i648B

	full_text
	
i64 %35
@bitcast8B3
1
	full_text$
"
 %37 = bitcast float* %36 to i32*
+float*8B

	full_text


float* %36
Hload8B>
<
	full_text/
-
+%38 = load i32, i32* %37, align 4, !tbaa !8
'i32*8B

	full_text


i32* %37
\getelementptr8BI
G
	full_text:
8
6%39 = getelementptr inbounds float, float* %4, i64 %15
%i648B

	full_text
	
i64 %15
@bitcast8B3
1
	full_text$
"
 %40 = bitcast float* %39 to i32*
+float*8B

	full_text


float* %39
Hstore8B=
;
	full_text.
,
*store i32 %38, i32* %40, align 4, !tbaa !8
%i328B

	full_text
	
i32 %38
'i32*8B

	full_text


i32* %40
\getelementptr8BI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %1, i64 %35
%i648B

	full_text
	
i64 %35
@bitcast8B3
1
	full_text$
"
 %42 = bitcast float* %41 to i32*
+float*8B

	full_text


float* %41
Hload8B>
<
	full_text/
-
+%43 = load i32, i32* %42, align 4, !tbaa !8
'i32*8B

	full_text


i32* %42
\getelementptr8BI
G
	full_text:
8
6%44 = getelementptr inbounds float, float* %5, i64 %15
%i648B

	full_text
	
i64 %15
@bitcast8B3
1
	full_text$
"
 %45 = bitcast float* %44 to i32*
+float*8B

	full_text


float* %44
Hstore8B=
;
	full_text.
,
*store i32 %43, i32* %45, align 4, !tbaa !8
%i328B

	full_text
	
i32 %43
'i32*8B

	full_text


i32* %45
'br8B

	full_text

br label %46
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 2) #4
$ret8B

	full_text


ret void
*float*8	B

	full_text

	float* %0
*float*8	B

	full_text

	float* %5
$i328	B

	full_text


i32 %7
*float*8	B

	full_text

	float* %3
*float*8	B

	full_text

	float* %2
*float*8	B

	full_text

	float* %1
*float*8	B

	full_text

	float* %4
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
#i648	B

	full_text	

i64 1
$i648	B

	full_text


i64 32
$i328	B

	full_text


i32 -1
#i328	B

	full_text	

i32 2
#i328	B

	full_text	

i32 0
#i648	B

	full_text	

i64 0       	
 		                   !    "# "$ "" %& %( '' )* )) +, +- .0 /1 // 23 22 45 44 67 66 89 88 :; :: <= << >? >@ >> AB AA CD CC EF EE GH GG IJ II KL KM KK NO PQ 4R GS S S S -T U V AW :    
	            !  # $" & (' *) ,' 0- 1/ 32 54 76 9 ;: =8 ?< @2 BA DC F HG JE LI M  O  - . /   'N O% % -+ -+ / XX YY P XX O YY OZ  [ 	[ \ )\ -] O^ ^ _ "
find_index_kernel"
_Z13get_global_idj"
_Z7barrierj*?
/rodinia-3.1-particlefilter-find_index_kernel.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

wgsize_log1p
@?A

transfer_bytes
???<
 
transfer_bytes_log1p
@?A

wgsize
?

devmap_label
 